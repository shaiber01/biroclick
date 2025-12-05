"""
Integration tests for the schema access validator.

These tests verify that the tool works correctly on the real codebase:
- Dynamic schema mapping discovery
- Real codebase validation
- Regression tests for known bugs

IMPORTANT: These are the ultimate tests. If these pass but bugs exist
in production, the unit tests are missing coverage.
"""

import json
import pytest
from pathlib import Path

from tools.validate_schema_access import (
    ViolationType,
    validate_agent_files,
    validate_prompts_module,
    build_agent_schema_mapping,
    validate_schema_naming_conventions,
    AGENT_OUTPUT_SCHEMA_MAPPING,
    SCHEMAS_DIR,
    AGENTS_DIR,
    PROJECT_ROOT,
    load_schema,
    extract_schema_fields,
    AgentSchemaInfo,
)

from .conftest import (
    validate_code_snippet,
    validate_code_with_mock_schema,
    assert_violation_found,
    assert_no_violations,
    MOCK_SCHEMA,
    TrackedVariable,
)


class TestDynamicSchemaMapping:
    """
    Tests for the dynamic agent-to-schema mapping discovery.
    """
    
    def test_mapping_is_populated(self):
        """Schema mapping should not be empty."""
        assert len(AGENT_OUTPUT_SCHEMA_MAPPING) > 0, \
            "No agent-to-schema mappings discovered"
    
    def test_discovers_known_agent_functions(self):
        """Should discover known agent functions."""
        # These are functions we know exist
        expected_functions = [
            ("planning.py", "adapt_prompts_node"),
            ("planning.py", "plan_node"),
            ("planning.py", "plan_reviewer_node"),
            ("code.py", "code_generator_node"),
            ("code.py", "code_reviewer_node"),
            ("design.py", "simulation_designer_node"),
            ("design.py", "design_reviewer_node"),
            ("execution.py", "execution_validator_node"),
            ("execution.py", "physics_sanity_node"),
            ("analysis.py", "results_analyzer_node"),
            ("reporting.py", "generate_report_node"),
        ]
        
        for file_name, func_name in expected_functions:
            assert file_name in AGENT_OUTPUT_SCHEMA_MAPPING, \
                f"File {file_name} not in mapping"
            assert func_name in AGENT_OUTPUT_SCHEMA_MAPPING[file_name], \
                f"Function {func_name} not found in {file_name} mapping"
    
    def test_extracts_variable_names(self):
        """Should extract the variable name used for LLM output."""
        for file_name, func_schemas in AGENT_OUTPUT_SCHEMA_MAPPING.items():
            for func_name, schema_info in func_schemas.items():
                assert isinstance(schema_info, AgentSchemaInfo), \
                    f"Expected AgentSchemaInfo for {file_name}:{func_name}"
                assert schema_info.variable_name, \
                    f"No variable name for {file_name}:{func_name}"
    
    def test_maps_to_existing_schemas(self):
        """All mapped schema files should exist."""
        missing_schemas = []
        
        for file_name, func_schemas in AGENT_OUTPUT_SCHEMA_MAPPING.items():
            for func_name, schema_info in func_schemas.items():
                schema_path = SCHEMAS_DIR / schema_info.schema_file
                if not schema_path.exists():
                    missing_schemas.append(
                        f"{file_name}:{func_name} -> {schema_info.schema_file}"
                    )
        
        assert not missing_schemas, \
            f"Missing schema files:\n  " + "\n  ".join(missing_schemas)
    
    def test_mapped_schemas_are_valid(self):
        """All mapped schemas should be valid JSON with required structure."""
        for file_name, func_schemas in AGENT_OUTPUT_SCHEMA_MAPPING.items():
            for func_name, schema_info in func_schemas.items():
                schema = load_schema(schema_info.schema_file)
                
                assert "properties" in schema, \
                    f"{schema_info.schema_file} missing 'properties'"
    
    def test_build_mapping_is_deterministic(self):
        """Building mapping twice should produce same result."""
        mapping1 = build_agent_schema_mapping()
        mapping2 = build_agent_schema_mapping()
        
        # Same files
        assert set(mapping1.keys()) == set(mapping2.keys())
        
        # Same functions per file
        for file_name in mapping1:
            assert set(mapping1[file_name].keys()) == set(mapping2[file_name].keys())


class TestRealCodebaseValidation:
    """
    Tests that run validation on the actual codebase.
    """
    
    def test_agent_files_pass_validation(self):
        """
        Current agent files should have 0 violations.
        
        If this test fails, it means real bugs exist in the codebase
        or the tool is over-flagging valid code.
        """
        result, _ = validate_agent_files()
        
        if result.violations:
            violation_list = "\n".join(
                f"  {v.file}:{v.line} [{v.type.value}] {v.message}"
                for v in result.violations
            )
            pytest.fail(
                f"Found {len(result.violations)} violation(s) in agent files:\n"
                f"{violation_list}"
            )
    
    def test_prompts_module_passes_validation(self):
        """
        Prompts module should have 0 violations.
        """
        result, _ = validate_prompts_module()
        
        if result.violations:
            violation_list = "\n".join(
                f"  {v.file}:{v.line} [{v.type.value}] {v.message}"
                for v in result.violations
            )
            pytest.fail(
                f"Found {len(result.violations)} violation(s) in prompts.py:\n"
                f"{violation_list}"
            )
    
    def test_validation_scans_all_agent_files(self):
        """Should scan all Python files in agents directory."""
        result, _ = validate_agent_files()
        
        # Count Python files in agents dir (excluding __init__.py and __pycache__)
        agent_files = list(AGENTS_DIR.rglob("*.py"))
        non_init_files = [f for f in agent_files 
                         if not f.name.startswith("__")]
        
        assert result.files_scanned >= len(non_init_files) * 0.8, \
            f"Expected to scan ~{len(non_init_files)} files, " \
            f"but only scanned {result.files_scanned}"
    
    def test_field_accesses_are_recorded(self):
        """Should record field accesses during validation."""
        result, _ = validate_agent_files()
        
        # We expect many field accesses across all agent files
        assert result.field_accesses, "No field accesses recorded"
        assert len(result.field_accesses) > 50, \
            f"Expected many field accesses, got {len(result.field_accesses)}"
        
        # Verify that accesses have proper structure
        for access in result.field_accesses:
            assert access.field, f"Field access missing field name: {access}"
            assert access.variable, f"Field access missing variable: {access}"
            assert access.access_type, f"Field access missing access_type: {access}"
        
        # Verify some common access types are present
        access_types = {a.access_type for a in result.field_accesses}
        assert "get" in access_types or "get_with_default" in access_types, \
            f"Expected .get() accesses, got types: {access_types}"


class TestSchemaNamingConvention:
    """
    Tests for schema naming convention enforcement.
    """
    
    def test_no_naming_convention_warnings(self):
        """All schemas should follow naming convention."""
        warnings = validate_schema_naming_conventions()
        
        if warnings:
            pytest.fail(
                f"Schema naming convention warnings:\n  " + 
                "\n  ".join(warnings)
            )
    
    def test_agent_schemas_follow_convention(self):
        """Agent output schemas should end with _output_schema.json."""
        # Check that mapped schemas follow convention
        for file_name, func_schemas in AGENT_OUTPUT_SCHEMA_MAPPING.items():
            for func_name, schema_info in func_schemas.items():
                assert schema_info.schema_file.endswith("_output_schema.json"), \
                    f"Schema {schema_info.schema_file} doesn't follow convention " \
                    f"(from {file_name}:{func_name})"


class TestRegressions:
    """
    Regression tests for specific bugs that were found and fixed.
    
    These tests ensure those bugs don't reappear.
    """
    
    def test_regression_derived_var_validation(self):
        """
        Regression: Derived variables must be validated against nested schemas.
        
        Previously, derived variables (e.g., `nested = agent_output.get("obj")`)
        were tracked but their field accesses weren't validated.
        """
        schema = {
            "properties": {
                "nested_obj": {
                    "type": "object",
                    "properties": {
                        "valid_field": {"type": "string"}
                    }
                }
            }
        }
        
        # Create temp schema file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_file = Path(f.name)
        
        try:
            code = '''
def test_func():
    nested = agent_output.get("nested_obj", {})
    value = nested.get("invalid_field")  # Should be caught!
'''
            result = validate_code_with_mock_schema(code, schema_file)
            
            # MUST catch invalid field on derived variable
            assert_violation_found(
                result,
                ViolationType.FIELD_NOT_IN_SCHEMA,
                field="invalid_field",
                variable="nested",
            )
        finally:
            schema_file.unlink()
    
    def test_regression_array_iteration_allowed(self):
        """
        Regression: Iteration over array-typed variables should be allowed.
        
        Previously, any iteration over tracked variables was flagged,
        even for array types where iteration is valid.
        """
        code = '''
def test_func():
    items = agent_output.get("array_field", [])
    for item in items:  # Should NOT be flagged
        pass
'''
        # Use mock schema with array field
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(MOCK_SCHEMA, f)
            schema_file = Path(f.name)
        
        try:
            result = validate_code_with_mock_schema(code, schema_file)
            
            # Should NOT have pattern violation for iteration
            pattern_violations = [v for v in result.violations 
                                 if v.type == ViolationType.PATTERN_NOT_WHITELISTED]
            assert not pattern_violations, \
                f"Array iteration incorrectly flagged: {pattern_violations}"
            
            # Should record the array_field access
            assert len(result.field_accesses) >= 1, "array_field access not recorded"
            fields = {a.field for a in result.field_accesses}
            assert "array_field" in fields, "array_field not in recorded fields"
        finally:
            schema_file.unlink()
    
    def test_regression_numeric_index_allowed(self):
        """
        Regression: Numeric index on arrays should be allowed.
        
        Previously, all subscript access was flagged, even array[0].
        """
        code = '''
def test_func():
    items = agent_output.get("array_field", [])
    if items:
        first = items[0]  # Should NOT be flagged
'''
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(MOCK_SCHEMA, f)
            schema_file = Path(f.name)
        
        try:
            result = validate_code_with_mock_schema(code, schema_file)
            
            # Should NOT have pattern violation for numeric index
            pattern_violations = [v for v in result.violations 
                                 if v.type == ViolationType.PATTERN_NOT_WHITELISTED]
            assert not pattern_violations, \
                f"Numeric index incorrectly flagged: {pattern_violations}"
            
            # Should record the array_field access
            assert len(result.field_accesses) >= 1, "array_field access not recorded"
        finally:
            schema_file.unlink()
    
    def test_regression_negative_index_allowed(self):
        """
        Regression: Negative index on arrays should be allowed.
        
        Previously, negative indices like items[-1] were flagged because
        Python AST represents them as UnaryOp(USub, Constant(1)).
        """
        code = '''
def test_func():
    items = agent_output.get("array_field", [])
    if items:
        last = items[-1]  # Should NOT be flagged
'''
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(MOCK_SCHEMA, f)
            schema_file = Path(f.name)
        
        try:
            result = validate_code_with_mock_schema(code, schema_file)
            
            # Should NOT have pattern violation for negative index
            pattern_violations = [v for v in result.violations 
                                 if v.type == ViolationType.PATTERN_NOT_WHITELISTED]
            assert not pattern_violations, \
                f"Negative index incorrectly flagged: {pattern_violations}"
        finally:
            schema_file.unlink()
    
    def test_regression_comprehension_vars_tracked(self):
        """
        Regression: Comprehension variables should be tracked.
        
        Previously, variables in list/dict/set comprehensions weren't tracked.
        """
        code = '''
def test_func():
    # x should be tracked, and invalid field should be caught
    values = [x.get("nonexistent") for x in agent_output.get("array_field", [])]
'''
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(MOCK_SCHEMA, f)
            schema_file = Path(f.name)
        
        try:
            result = validate_code_with_mock_schema(code, schema_file)
            
            # MUST catch invalid field on comprehension variable
            assert_violation_found(
                result,
                ViolationType.FIELD_NOT_IN_SCHEMA,
                field="nonexistent",
                variable="x",
            )
        finally:
            schema_file.unlink()


class TestToolCorrectness:
    """
    Meta-tests that verify the tool itself is working correctly.
    """
    
    def test_tool_catches_known_invalid_pattern(self):
        """Tool must catch an intentionally invalid pattern."""
        code = '''
def test_func():
    # This is definitely wrong - using subscript
    value = agent_output["field"]
'''
        result = validate_code_snippet(code)
        
        # MUST catch this
        assert result.violations, "Tool failed to catch subscript access"
        assert any(v.type == ViolationType.PATTERN_NOT_WHITELISTED 
                  for v in result.violations)
        
        # Verify violation details
        violation = next(v for v in result.violations 
                        if v.type == ViolationType.PATTERN_NOT_WHITELISTED)
        assert violation.variable == "agent_output", \
            f"Expected variable 'agent_output', got '{violation.variable}'"
        assert "Subscript" in violation.message, \
            f"Expected 'Subscript' in message, got '{violation.message}'"
    
    def test_tool_catches_known_invalid_field(self):
        """Tool must catch access to non-existent field."""
        schema = {
            "properties": {
                "valid_field": {"type": "string"}
            }
        }
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_file = Path(f.name)
        
        try:
            code = '''
def test_func():
    value = agent_output.get("completely_invalid_field_name")
'''
            result = validate_code_with_mock_schema(code, schema_file)
            
            # MUST catch this
            assert result.violations, "Tool failed to catch invalid field"
            assert any(v.type == ViolationType.FIELD_NOT_IN_SCHEMA 
                      for v in result.violations)
            
            # Verify violation details
            violation = next(v for v in result.violations 
                            if v.type == ViolationType.FIELD_NOT_IN_SCHEMA)
            assert violation.field == "completely_invalid_field_name", \
                f"Expected field 'completely_invalid_field_name', got '{violation.field}'"
            assert violation.variable == "agent_output", \
                f"Expected variable 'agent_output', got '{violation.variable}'"
        finally:
            schema_file.unlink()
    
    def test_tool_allows_known_valid_pattern(self):
        """Tool must NOT flag valid .get() pattern."""
        schema = {
            "properties": {
                "valid_field": {"type": "string"}
            }
        }
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_file = Path(f.name)
        
        try:
            code = '''
def test_func():
    value = agent_output.get("valid_field")
'''
            result = validate_code_with_mock_schema(code, schema_file)
            
            # Must NOT have violations
            assert not result.violations, \
                f"Tool incorrectly flagged valid pattern: {result.violations}"
            
            # Must have recorded the access
            assert len(result.field_accesses) == 1, \
                f"Expected 1 field access, got {len(result.field_accesses)}"
            assert result.field_accesses[0].field == "valid_field", \
                f"Expected field 'valid_field', got '{result.field_accesses[0].field}'"
            assert result.field_accesses[0].variable == "agent_output", \
                f"Expected variable 'agent_output', got '{result.field_accesses[0].variable}'"
        finally:
            schema_file.unlink()
