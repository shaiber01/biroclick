"""
Unit tests for pairing rule validation.

These tests verify that check_pairing_violations correctly:
1. Accepts valid pairings
2. Rejects invalid pairings
3. Handles edge cases (None values, variable assignments)
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validate_trigger_pairing import (
    analyze_source,
    check_pairing_violations,
    ViolationType,
)


class TestValidPairings:
    """Tests that valid code does not produce violations."""
    
    def test_dict_literal_with_both_keys_is_valid(self, valid_dict_literal):
        """Dict literal with both trigger and questions should be valid."""
        result = analyze_source(valid_dict_literal, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0
    
    def test_dict_literal_multikey_is_valid(self, valid_dict_literal_multikey):
        """Dict with many keys including both trigger and questions should be valid."""
        result = analyze_source(valid_dict_literal_multikey, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0
    
    def test_subscript_with_nearby_questions_is_valid(self, valid_subscript):
        """Subscript assignment with questions nearby should be valid."""
        result = analyze_source(valid_subscript, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0
    
    def test_subscript_within_10_lines_is_valid(self, valid_subscript_nearby):
        """Subscript with questions within 10 lines should be valid."""
        result = analyze_source(valid_subscript_nearby, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0
    
    def test_clearing_with_none_is_valid(self, clearing_trigger):
        """Clearing trigger with None should be valid without questions."""
        result = analyze_source(clearing_trigger, "test.py")
        violations = check_pairing_violations(result)
        
        # Should have no violations at all
        assert len(violations) == 0
    
    def test_clearing_none_alone_is_valid(self, clearing_trigger_alone):
        """Clearing trigger with None alone should be valid."""
        result = analyze_source(clearing_trigger_alone, "test.py")
        violations = check_pairing_violations(result)
        
        assert len(violations) == 0
    
    def test_preserving_known_variable_is_valid(self, preserving_trigger):
        """Preserving trigger via known variable name should be valid."""
        result = analyze_source(preserving_trigger, "test.py")
        violations = check_pairing_violations(result)
        
        # Should have no errors (might have warnings)
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0


class TestInvalidPairings:
    """Tests that invalid code produces appropriate violations."""
    
    def test_dict_literal_without_questions_is_violation(self, invalid_dict_literal):
        """Dict literal missing questions should be a violation."""
        result = analyze_source(invalid_dict_literal, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 1
        assert errors[0].type == ViolationType.UNPAIRED_IN_DICT
    
    def test_dict_literal_empty_questions_is_violation(self, invalid_dict_literal_empty_questions):
        """Dict with trigger but no questions key should be a violation."""
        result = analyze_source(invalid_dict_literal_empty_questions, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 1
        assert errors[0].type == ViolationType.UNPAIRED_IN_DICT
    
    def test_subscript_without_nearby_questions_is_violation(self, invalid_subscript_no_questions):
        """Subscript without questions nearby should be a violation."""
        result = analyze_source(invalid_subscript_no_questions, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 1
        assert errors[0].type == ViolationType.UNPAIRED_SUBSCRIPT
    
    def test_subscript_with_far_questions_is_violation(self, invalid_subscript_far_questions):
        """Subscript with questions more than 10 lines away should be a violation."""
        result = analyze_source(invalid_subscript_far_questions, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 1
        assert errors[0].type == ViolationType.UNPAIRED_SUBSCRIPT
    
    def test_invalid_nested_dict_is_violation(self, invalid_nested_dict):
        """Nested dict without questions should be a violation."""
        result = analyze_source(invalid_nested_dict, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 1
        assert errors[0].type == ViolationType.UNPAIRED_IN_DICT


class TestSuspiciousVariables:
    """Tests for handling suspicious variable assignments."""
    
    def test_unknown_variable_is_warning(self, suspicious_variable):
        """Unknown variable assignment should produce a warning."""
        result = analyze_source(suspicious_variable, "test.py")
        violations = check_pairing_violations(result)
        
        warnings = [v for v in violations if v.severity == "warning"]
        assert len(warnings) == 1
        assert warnings[0].type == ViolationType.SUSPICIOUS_VARIABLE
    
    def test_unknown_variable_is_not_error(self, suspicious_variable):
        """Unknown variable should be warning, not error."""
        result = analyze_source(suspicious_variable, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0
    
    def test_trigger_variable_is_allowed(self, preserving_trigger_param):
        """Variable named 'trigger' should be treated as preservation."""
        result = analyze_source(preserving_trigger_param, "test.py")
        violations = check_pairing_violations(result)
        
        # Should be a warning because 'trigger' is a known preservation var
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0


class TestViolationDetails:
    """Tests that violations contain appropriate details."""
    
    def test_violation_contains_filepath(self, invalid_dict_literal):
        """Violation should contain the filepath."""
        result = analyze_source(invalid_dict_literal, "custom/path.py")
        violations = check_pairing_violations(result)
        
        assert violations[0].filepath == "custom/path.py"
    
    def test_violation_contains_line_number(self, invalid_dict_literal):
        """Violation should contain line number."""
        result = analyze_source(invalid_dict_literal, "test.py")
        violations = check_pairing_violations(result)
        
        assert violations[0].line > 0
    
    def test_violation_contains_function_name(self, invalid_dict_literal):
        """Violation should contain function name."""
        result = analyze_source(invalid_dict_literal, "test.py")
        violations = check_pairing_violations(result)
        
        assert violations[0].function_name == "broken_handler"
    
    def test_violation_contains_trigger_value(self, invalid_dict_literal):
        """Violation message should contain the trigger value."""
        result = analyze_source(invalid_dict_literal, "test.py")
        violations = check_pairing_violations(result)
        
        assert "test_trigger" in violations[0].message


class TestMultipleTriggers:
    """Tests for code with multiple trigger assignments."""
    
    def test_multiple_valid_triggers_no_violations(self, multiple_triggers_same_function):
        """Multiple properly paired triggers should produce no errors."""
        result = analyze_source(multiple_triggers_same_function, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0
    
    def test_mixed_valid_invalid_produces_violations(self):
        """Code with both valid and invalid triggers should flag only invalid."""
        # The orphan trigger must be more than 10 lines away from any questions
        code = '''
def mixed_handler(condition):
    result = {}
    
    if condition:
        # Valid - has questions
        result["ask_user_trigger"] = "valid_trigger"
        result["pending_user_questions"] = ["Valid question"]
        return result
    
    # Many lines of unrelated code to create distance
    x = 1
    y = 2
    z = 3
    a = 4
    b = 5
    c = 6
    d = 7
    e = 8
    f = 9
    g = 10
    h = 11
    
    # Invalid - missing questions and more than 10 lines from valid questions
    result["ask_user_trigger"] = "orphan_trigger"
    return result
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 1
        assert "orphan_trigger" in errors[0].message


class TestEdgeCases:
    """Tests for edge cases in pairing validation."""
    
    def test_empty_result_no_violations(self, no_triggers):
        """Code without triggers should have no violations."""
        result = analyze_source(no_triggers, "test.py")
        violations = check_pairing_violations(result)
        
        assert len(violations) == 0
    
    def test_parse_error_no_violations(self, syntax_error_code):
        """Code with parse error should have no violations (can't analyze)."""
        result = analyze_source(syntax_error_code, "test.py")
        violations = check_pairing_violations(result)
        
        # No violations because no assignments were found
        assert len(violations) == 0
    
    def test_class_method_detection(self, class_method_trigger):
        """Triggers in class methods should be validated."""
        result = analyze_source(class_method_trigger, "test.py")
        violations = check_pairing_violations(result)
        
        # Should be valid (has questions)
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0
    
    def test_async_function_detection(self, async_function_trigger):
        """Triggers in async functions should be validated."""
        result = analyze_source(async_function_trigger, "test.py")
        violations = check_pairing_violations(result)
        
        # Should be valid (has questions)
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0
