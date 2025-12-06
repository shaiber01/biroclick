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
        
        # Should have no ERRORS (warnings for empty questions list are OK in clearing scenario)
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0, f"Clearing should have no errors: {errors}"
    
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
        # Also verify the analysis found no triggers
        assert len(result.trigger_assignments) == 0, "Should find no triggers in no_triggers fixture"
    
    def test_parse_error_no_violations(self, syntax_error_code):
        """Code with parse error should have no violations (can't analyze)."""
        result = analyze_source(syntax_error_code, "test.py")
        violations = check_pairing_violations(result)
        
        # No violations because no assignments were found
        assert len(violations) == 0
        # Verify parse error was recorded
        assert len(result.parse_errors) > 0, "Should record the parse error"
    
    def test_class_method_detection(self, class_method_trigger):
        """Triggers in class methods should be validated."""
        result = analyze_source(class_method_trigger, "test.py")
        violations = check_pairing_violations(result)
        
        # Should be valid (has questions)
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0
        # Verify method name was tracked
        assert result.trigger_assignments[0].function_name == "handle_error"
    
    def test_async_function_detection(self, async_function_trigger):
        """Triggers in async functions should be validated."""
        result = analyze_source(async_function_trigger, "test.py")
        violations = check_pairing_violations(result)
        
        # Should be valid (has questions)
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0
        # Verify async function name was tracked
        assert result.trigger_assignments[0].function_name == "async_handler"


class TestBoundaryConditions:
    """
    Tests for the 10-line proximity rule boundary conditions.
    
    These tests verify the exact boundary behavior:
    - 9 lines apart: VALID
    - 10 lines apart: VALID (inclusive boundary)
    - 11 lines apart: INVALID
    """
    
    def test_subscript_exactly_9_lines_apart_is_valid(self):
        """Questions exactly 9 lines after trigger should be valid."""
        # Line 1: def, Line 2: result={}, Line 3: trigger, Lines 4-11: filler, Line 12: questions
        # Distance: 12 - 3 = 9 lines
        code = '''
def handler():
    result = {}
    result["ask_user_trigger"] = "test"
    x = 1
    x = 2
    x = 3
    x = 4
    x = 5
    x = 6
    x = 7
    result["pending_user_questions"] = ["Q"]
    return result
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0, f"9 lines apart should be valid, but got: {errors}"
    
    def test_subscript_exactly_10_lines_apart_is_valid(self):
        """Questions exactly 10 lines after trigger should be valid (boundary)."""
        # Line 3: trigger, Line 13: questions = 10 lines apart
        code = '''
def handler():
    result = {}
    result["ask_user_trigger"] = "test"
    x = 1
    x = 2
    x = 3
    x = 4
    x = 5
    x = 6
    x = 7
    x = 8
    result["pending_user_questions"] = ["Q"]
    return result
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0, f"10 lines apart should be valid, but got: {errors}"
    
    def test_subscript_exactly_11_lines_apart_is_invalid(self):
        """Questions exactly 11 lines after trigger should be INVALID."""
        # Line 4: trigger, Line 15: questions = 11 lines apart
        code = '''
def handler():
    result = {}
    result["ask_user_trigger"] = "test"
    x = 1
    x = 2
    x = 3
    x = 4
    x = 5
    x = 6
    x = 7
    x = 8
    x = 9
    x = 10
    result["pending_user_questions"] = ["Q"]
    return result
'''
        result = analyze_source(code, "test.py")
        
        # Verify the actual lines to catch test bugs
        assert len(result.trigger_assignments) == 1, "Should find exactly 1 trigger"
        trigger_line = result.trigger_assignments[0].line
        questions_lines = list(result.questions_lines)
        assert len(questions_lines) == 1, "Should find exactly 1 questions line"
        questions_line = questions_lines[0]
        actual_distance = abs(trigger_line - questions_line)
        assert actual_distance == 11, f"Test setup error: expected 11 lines apart, got {actual_distance}"
        
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 1, f"11 lines apart should be invalid, but got {len(errors)} errors"
        assert errors[0].type == ViolationType.UNPAIRED_SUBSCRIPT
    
    def test_subscript_questions_before_trigger_within_10_is_valid(self):
        """Questions 10 lines BEFORE trigger should also be valid."""
        # Line 3: questions, Line 13: trigger = 10 lines apart (questions first)
        code = '''
def handler():
    result = {}
    result["pending_user_questions"] = ["Q"]
    x = 1
    x = 2
    x = 3
    x = 4
    x = 5
    x = 6
    x = 7
    x = 8
    result["ask_user_trigger"] = "test"
    return result
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0, f"Questions before trigger (10 lines) should be valid: {errors}"


class TestFalseNegativeDetection:
    """
    Tests that verify the tool correctly catches known buggy patterns.
    
    These tests inject KNOWN BUGS and verify they are detected.
    If these tests pass without detecting violations, the tool has a bug.
    """
    
    def test_catches_dict_literal_with_only_trigger(self):
        """Tool MUST catch dict literal that only has trigger, no questions."""
        code = '''
def buggy_handler():
    return {
        "ask_user_trigger": "orphan_trigger",
        "some_other_key": "value",
    }
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 1, "MUST detect missing questions in dict literal"
        assert errors[0].type == ViolationType.UNPAIRED_IN_DICT
        assert "orphan_trigger" in errors[0].message
    
    def test_catches_subscript_without_any_questions(self):
        """Tool MUST catch subscript trigger with no questions anywhere."""
        code = '''
def buggy_handler():
    result = {}
    result["ask_user_trigger"] = "lonely_trigger"
    result["workflow_phase"] = "stuck"
    return result
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 1, "MUST detect missing questions for subscript"
        assert errors[0].type == ViolationType.UNPAIRED_SUBSCRIPT
        assert "lonely_trigger" in errors[0].message
    
    def test_catches_multiple_unpaired_triggers(self):
        """Tool MUST catch ALL unpaired triggers, not just the first one."""
        code = '''
def multi_bug():
    result = {}
    result["ask_user_trigger"] = "bug1"
    result["workflow_phase"] = "x"
    return result

def another_bug():
    return {
        "ask_user_trigger": "bug2",
    }
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 2, f"MUST detect BOTH unpaired triggers, found {len(errors)}"
        
        error_messages = " ".join(e.message for e in errors)
        assert "bug1" in error_messages, "Should catch bug1"
        assert "bug2" in error_messages, "Should catch bug2"
    
    def test_detects_empty_questions_list_in_dict(self):
        """
        A dict with trigger and EMPTY questions list should be detected.
        Empty questions = no questions to show user = still a bug.
        """
        code = '''
def empty_questions_bug():
    return {
        "ask_user_trigger": "trigger_with_empty_questions",
        "pending_user_questions": [],  # Empty! Still a bug.
    }
'''
        result = analyze_source(code, "test.py")
        
        # Verify the trigger is detected and marked as NOT paired (empty questions don't count)
        assert len(result.trigger_assignments) == 1, "Should detect the trigger"
        assignment = result.trigger_assignments[0]
        assert assignment.paired_in_dict is False, (
            "Empty questions list should NOT count as paired"
        )
        
        violations = check_pairing_violations(result)
        
        # Should have UNPAIRED_IN_DICT error for the trigger
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 1, f"Should detect unpaired trigger, got: {errors}"
        assert errors[0].type == ViolationType.UNPAIRED_IN_DICT
        
        # Should also have EMPTY_QUESTIONS warning
        warnings = [v for v in violations if v.severity == "warning"]
        empty_warnings = [v for v in warnings if v.type == ViolationType.EMPTY_QUESTIONS]
        assert len(empty_warnings) == 1, "Should warn about empty questions list"


class TestCrossFunctionBoundary:
    """
    Tests that verify the tool respects function boundaries.
    
    A trigger in function A should NOT be "paired" with questions in function B,
    even if they happen to be within 10 lines of each other.
    """
    
    def test_cross_function_not_paired(self):
        """
        Questions in one function should NOT pair with trigger in another function.
        
        This was a bug: the tool used to check ALL questions in the file,
        not just questions in the same function.
        """
        code = '''
def func_a():
    result = {}
    result["pending_user_questions"] = ["Question from A"]
    return result

def func_b():
    result = {}
    result["ask_user_trigger"] = "orphan_trigger"
    return result
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        # The trigger in func_b should be flagged as unpaired
        # even though questions in func_a are nearby
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 1, (
            f"Trigger in func_b should be unpaired (questions are in func_a). Got: {errors}"
        )
        assert errors[0].type == ViolationType.UNPAIRED_SUBSCRIPT
        assert "func_b" in errors[0].message, "Error should mention the function name"
    
    def test_same_function_still_pairs(self):
        """Questions in same function should still pair correctly."""
        code = '''
def func_a():
    result = {}
    result["pending_user_questions"] = ["Question"]
    return result

def func_b():
    result = {}
    result["ask_user_trigger"] = "trigger"
    result["pending_user_questions"] = ["Proper question"]
    return result
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        # func_b trigger should be paired with func_b questions
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0, f"func_b trigger should be properly paired. Got: {errors}"
    
    def test_global_scope_pairing(self):
        """Global scope triggers should pair with global scope questions."""
        code = '''
result = {}
result["ask_user_trigger"] = "global_trigger"
result["pending_user_questions"] = ["Global question"]
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0, f"Global trigger should pair with global questions. Got: {errors}"
    
    def test_global_trigger_not_paired_with_function_questions(self):
        """Global scope trigger should NOT pair with questions inside a function."""
        code = '''
def some_func():
    result = {}
    result["pending_user_questions"] = ["Question in function"]
    return result

result = {}
result["ask_user_trigger"] = "global_orphan"
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 1, (
            f"Global trigger should NOT pair with function questions. Got: {errors}"
        )


class TestClearingPatternSuppression:
    """
    Tests that verify clearing patterns (trigger=None + questions=[]) suppress warnings.
    
    A clearing pattern is when BOTH:
    - ask_user_trigger is set to None
    - pending_user_questions is set to []
    IN THE SAME DICT LITERAL.
    
    This is a legitimate pattern for resetting state, not a bug.
    """
    
    def test_clearing_pattern_in_dict_suppresses_warning(self):
        """
        Dict with trigger=None AND questions=[] should NOT produce a warning.
        This is the canonical clearing pattern used in supervisor_node.
        """
        code = '''
def clear_user_state():
    return {
        "ask_user_trigger": None,
        "pending_user_questions": [],
        "user_responses": {},
    }
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        # Should have NO violations at all - this is intentional clearing
        assert len(violations) == 0, (
            f"Clearing pattern should not produce warnings. Got: {violations}"
        )
        
        # Verify the questions assignment was detected as clearing pattern
        assert len(result.questions_assignments) == 1
        assert result.questions_assignments[0].is_clearing_pattern is True, (
            "Questions assignment should be marked as clearing pattern"
        )
    
    def test_empty_questions_without_trigger_none_still_warns(self):
        """
        Dict with ONLY questions=[] (no trigger) should still warn.
        This is NOT a clearing pattern.
        """
        code = '''
def suspicious_empty():
    return {
        "workflow_phase": "somewhere",
        "pending_user_questions": [],  # No trigger in this dict!
    }
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        # Should have a warning for empty questions
        warnings = [v for v in violations if v.severity == "warning"]
        assert len(warnings) == 1, (
            f"Empty questions without trigger=None should warn. Got: {warnings}"
        )
        assert warnings[0].type == ViolationType.EMPTY_QUESTIONS
        
        # Verify NOT marked as clearing pattern
        assert result.questions_assignments[0].is_clearing_pattern is False
    
    def test_empty_questions_with_non_none_trigger_still_warns(self):
        """
        Dict with trigger="value" AND questions=[] should warn.
        Setting a real trigger with no questions is a BUG, not clearing.
        """
        code = '''
def buggy_trigger():
    return {
        "ask_user_trigger": "some_trigger",
        "pending_user_questions": [],  # BUG: trigger set but no questions!
    }
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        # Should have BOTH:
        # 1. Error for unpaired trigger in dict
        # 2. Warning for empty questions
        errors = [v for v in violations if v.severity == "error"]
        warnings = [v for v in violations if v.severity == "warning"]
        
        assert len(errors) == 1, (
            f"Trigger with empty questions should be an error. Got: {errors}"
        )
        assert errors[0].type == ViolationType.UNPAIRED_IN_DICT
        
        assert len(warnings) == 1, (
            f"Empty questions should also warn. Got: {warnings}"
        )
        assert warnings[0].type == ViolationType.EMPTY_QUESTIONS
        
        # NOT a clearing pattern
        assert result.questions_assignments[0].is_clearing_pattern is False
    
    def test_subscript_clearing_within_5_lines_suppresses(self):
        """
        Subscript assignments with trigger=None nearby (within ±5 lines)
        should be detected as a clearing pattern and NOT warn.
        
        This handles code like:
            result["ask_user_trigger"] = None
            result["pending_user_questions"] = []
        """
        code = '''
def subscript_clearing():
    result = {}
    result["ask_user_trigger"] = None
    result["pending_user_questions"] = []
    return result
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        # Subscript clearing pattern should NOT warn
        warnings = [v for v in violations if v.severity == "warning"]
        assert len(warnings) == 0, (
            f"Subscript clearing pattern should not warn. Got: {warnings}"
        )
    
    def test_subscript_clearing_beyond_5_lines_still_warns(self):
        """
        Subscript assignments with trigger=None too far away (>5 lines)
        should still warn - not a recognized clearing pattern.
        """
        code = '''
def not_clearing():
    result = {}
    result["ask_user_trigger"] = None
    # Line 1
    # Line 2
    # Line 3
    # Line 4
    # Line 5
    # Line 6 - now trigger is more than 5 lines away
    result["pending_user_questions"] = []
    return result
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        # Should still warn - trigger=None is too far away
        warnings = [v for v in violations if v.severity == "warning"]
        assert len(warnings) == 1, (
            f"Subscript empty questions far from trigger=None should warn. Got: {warnings}"
        )
    
    def test_subscript_clearing_different_function_still_warns(self):
        """
        trigger=None in one function should NOT suppress questions=[] in another.
        """
        code = '''
def func_a():
    result = {}
    result["ask_user_trigger"] = None
    return result

def func_b():
    result = {}
    result["pending_user_questions"] = []  # Should warn
    return result
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        # Should warn - trigger=None is in different function
        warnings = [v for v in violations if v.severity == "warning"]
        assert len(warnings) == 1, (
            f"Empty questions with trigger=None in different function should warn. Got: {warnings}"
        )
    
    def test_multiple_dicts_only_suppresses_clearing_ones(self):
        """
        When file has multiple dicts, only the clearing patterns suppress.
        Non-clearing empty questions should still warn.
        """
        code = '''
def mixed_patterns():
    # This is a clearing pattern - should NOT warn
    clearing = {
        "ask_user_trigger": None,
        "pending_user_questions": [],
    }
    
    # This is NOT clearing - should warn
    suspicious = {
        "workflow_phase": "test",
        "pending_user_questions": [],
    }
    
    return clearing, suspicious
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        # Should have exactly 1 warning (for suspicious, not clearing)
        warnings = [v for v in violations if v.severity == "warning"]
        assert len(warnings) == 1, (
            f"Should have exactly 1 warning for non-clearing empty questions. Got: {warnings}"
        )
        
        # Verify we have 2 questions assignments, one clearing, one not
        assert len(result.questions_assignments) == 2
        clearing_count = sum(1 for q in result.questions_assignments if q.is_clearing_pattern)
        assert clearing_count == 1, (
            f"Should have exactly 1 clearing pattern. Got: {clearing_count}"
        )
    
    def test_clearing_with_additional_keys_still_suppresses(self):
        """
        Clearing pattern with many other keys should still suppress.
        Real-world clearing dicts often have other fields.
        """
        code = '''
def full_clearing():
    return {
        "workflow_phase": "supervision",
        "ask_user_trigger": None,
        "pending_user_questions": [],
        "user_responses": {},
        "supervisor_verdict": "ok_continue",
        "supervisor_feedback": "Auto-recovered",
    }
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        # Should have no violations
        assert len(violations) == 0, (
            f"Full clearing dict should not warn. Got: {violations}"
        )
        assert result.questions_assignments[0].is_clearing_pattern is True
    
    def test_clearing_pattern_detection_is_per_dict(self):
        """
        Clearing detection must be per-dict, not per-function.
        trigger=None in one dict doesn't suppress questions=[] in another.
        """
        code = '''
def two_returns(flag):
    if flag:
        # This dict has trigger=None but NO questions
        return {"ask_user_trigger": None}
    else:
        # This dict has questions=[] but NO trigger
        return {"pending_user_questions": []}  # Should warn!
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        # The second dict should warn (empty questions, not a clearing pattern)
        warnings = [v for v in violations if v.severity == "warning"]
        assert len(warnings) == 1, (
            f"Second dict should warn. Got: {warnings}"
        )
        
        # First questions assignment is NOT a clearing pattern
        # (it's in a different dict than trigger=None)
        for q in result.questions_assignments:
            assert q.is_clearing_pattern is False, (
                "Questions in separate dict from trigger should not be clearing pattern"
            )
    
    def test_trigger_none_alone_without_questions_is_valid(self):
        """
        trigger=None alone (no questions key at all) is valid.
        This is clearing the trigger without touching questions.
        """
        code = '''
def clear_trigger_only():
    return {
        "ask_user_trigger": None,
        # No questions key - this is fine
    }
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        # Should have no violations
        assert len(violations) == 0, (
            f"trigger=None alone should be valid. Got: {violations}"
        )
    
    def test_update_call_clearing_pattern_suppresses(self):
        """
        Clearing pattern via state.update() should also suppress.
        """
        code = '''
def update_clearing():
    state.update({
        "ask_user_trigger": None,
        "pending_user_questions": [],
    })
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        # Should have no violations (clearing pattern via update)
        assert len(violations) == 0, (
            f"Clearing via update() should not warn. Got: {violations}"
        )
        assert result.questions_assignments[0].is_clearing_pattern is True


class TestValueTypeClassification:
    """
    Tests that verify different value types are correctly classified.
    
    The tool should distinguish between:
    - None (clearing) - VALID without questions
    - string_literal - Requires questions
    - variable - Depends on variable name
    - other (f-strings, calls, etc.) - Should be flagged
    """
    
    def test_fstring_value_is_classified_as_other(self):
        """F-string trigger values should be classified as 'other'."""
        code = '''
def dynamic_trigger(prefix):
    return {
        "ask_user_trigger": f"{prefix}_trigger",
        "pending_user_questions": ["Question"],
    }
'''
        result = analyze_source(code, "test.py")
        
        assert len(result.trigger_assignments) == 1
        # F-strings become JoinedStr in AST, should be classified as "other"
        # This is paired in dict so should be valid, but verify classification
        assignment = result.trigger_assignments[0]
        assert assignment.value_type in ("other", "string_literal"), (
            f"F-string should be 'other' or detected as string, got: {assignment.value_type}"
        )
    
    def test_function_call_value_is_classified_as_other(self):
        """Function call as trigger value should be classified as 'other'."""
        code = '''
def computed_trigger():
    return {
        "ask_user_trigger": get_trigger_name(),
        "pending_user_questions": ["Question"],
    }
'''
        result = analyze_source(code, "test.py")
        
        assert len(result.trigger_assignments) == 1
        assignment = result.trigger_assignments[0]
        assert assignment.value_type == "other", (
            f"Function call should be 'other', got: {assignment.value_type}"
        )
    
    def test_conditional_value_is_classified(self):
        """Conditional expression as value should be handled."""
        code = '''
def conditional_trigger(flag):
    return {
        "ask_user_trigger": "trigger_a" if flag else "trigger_b",
        "pending_user_questions": ["Question"],
    }
'''
        result = analyze_source(code, "test.py")
        
        assert len(result.trigger_assignments) == 1
        # Conditional (ternary) is IfExp in AST, should be "other"
        assignment = result.trigger_assignments[0]
        assert assignment.value_type == "other", (
            f"Conditional should be 'other', got: {assignment.value_type}"
        )
