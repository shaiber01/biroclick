"""
Unit tests for the TriggerPairingAnalyzer AST visitor.

These tests verify that the analyzer correctly:
1. Detects trigger assignments in dict literals
2. Detects trigger assignments via subscripts
3. Tracks function context
4. Handles edge cases gracefully
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validate_trigger_pairing import (
    analyze_source,
    AnalysisResult,
    check_pairing_violations,
    ViolationType,
)


class TestDictLiteralDetection:
    """Tests for detecting triggers in dict literals."""
    
    def test_detects_trigger_in_dict_literal(self, valid_dict_literal):
        """Should detect trigger assignment in dict literal."""
        result = analyze_source(valid_dict_literal, "test.py")
        
        assert len(result.trigger_assignments) == 1
        assignment = result.trigger_assignments[0]
        assert assignment.in_dict_literal is True
        assert assignment.value_type == "string_literal"
        assert assignment.value == "test_trigger"
    
    def test_detects_paired_dict_literal(self, valid_dict_literal):
        """Should detect that dict literal has both trigger and questions."""
        result = analyze_source(valid_dict_literal, "test.py")
        
        assert len(result.trigger_assignments) == 1
        assignment = result.trigger_assignments[0]
        assert assignment.paired_in_dict is True
    
    def test_detects_unpaired_dict_literal(self, invalid_dict_literal):
        """Should detect that dict literal is missing questions."""
        result = analyze_source(invalid_dict_literal, "test.py")
        
        assert len(result.trigger_assignments) == 1
        assignment = result.trigger_assignments[0]
        assert assignment.in_dict_literal is True
        assert assignment.paired_in_dict is False
    
    def test_detects_nested_dict_trigger(self, nested_dict_literal):
        """Should detect trigger in nested dict."""
        result = analyze_source(nested_dict_literal, "test.py")
        
        assert len(result.trigger_assignments) == 1
        assignment = result.trigger_assignments[0]
        assert assignment.in_dict_literal is True
        assert assignment.paired_in_dict is True
    
    def test_detects_invalid_nested_dict(self, invalid_nested_dict):
        """Should detect unpaired trigger in nested dict."""
        result = analyze_source(invalid_nested_dict, "test.py")
        
        assert len(result.trigger_assignments) == 1
        assignment = result.trigger_assignments[0]
        assert assignment.paired_in_dict is False


class TestSubscriptDetection:
    """Tests for detecting triggers via subscript assignments."""
    
    def test_detects_subscript_assignment(self, valid_subscript):
        """Should detect trigger in subscript assignment."""
        result = analyze_source(valid_subscript, "test.py")
        
        assert len(result.trigger_assignments) == 1
        assignment = result.trigger_assignments[0]
        assert assignment.in_dict_literal is False
        assert assignment.value_type == "string_literal"
        assert assignment.value == "physics_failure_limit"
    
    def test_detects_questions_line_for_subscript(self, valid_subscript):
        """Should track line where questions are assigned via subscript."""
        result = analyze_source(valid_subscript, "test.py")
        
        assert len(result.questions_lines) > 0
    
    def test_subscript_not_marked_as_dict_literal(self, valid_subscript):
        """Subscript assignment should not be marked as dict literal."""
        result = analyze_source(valid_subscript, "test.py")
        
        assignment = result.trigger_assignments[0]
        assert assignment.in_dict_literal is False
        assert assignment.paired_in_dict is False  # N/A for subscripts


class TestValueClassification:
    """Tests for classifying assigned values."""
    
    def test_classifies_none_value(self, clearing_trigger):
        """Should classify None assignment as 'none' type."""
        result = analyze_source(clearing_trigger, "test.py")
        
        assert len(result.trigger_assignments) == 1
        assignment = result.trigger_assignments[0]
        assert assignment.value_type == "none"
        assert assignment.value is None
    
    def test_classifies_string_literal(self, valid_dict_literal):
        """Should classify string literal assignment."""
        result = analyze_source(valid_dict_literal, "test.py")
        
        assignment = result.trigger_assignments[0]
        assert assignment.value_type == "string_literal"
        assert assignment.value == "test_trigger"
    
    def test_classifies_variable(self, preserving_trigger):
        """Should classify variable assignment."""
        result = analyze_source(preserving_trigger, "test.py")
        
        assert len(result.trigger_assignments) == 1
        assignment = result.trigger_assignments[0]
        assert assignment.value_type == "variable"
        assert assignment.value == "ask_user_trigger"


class TestFunctionContext:
    """Tests for tracking function context."""
    
    def test_tracks_function_name(self, valid_dict_literal):
        """Should track which function the trigger is in."""
        result = analyze_source(valid_dict_literal, "test.py")
        
        assignment = result.trigger_assignments[0]
        assert assignment.function_name == "handle_limit"
    
    def test_tracks_class_method_name(self, class_method_trigger):
        """Should track method name for class methods."""
        result = analyze_source(class_method_trigger, "test.py")
        
        assignment = result.trigger_assignments[0]
        assert assignment.function_name == "handle_error"
    
    def test_tracks_async_function_name(self, async_function_trigger):
        """Should track async function name."""
        result = analyze_source(async_function_trigger, "test.py")
        
        assignment = result.trigger_assignments[0]
        assert assignment.function_name == "async_handler"


class TestMultipleAssignments:
    """Tests for handling multiple trigger assignments."""
    
    def test_detects_multiple_triggers(self, multiple_triggers_same_function):
        """Should detect all trigger assignments in a function."""
        result = analyze_source(multiple_triggers_same_function, "test.py")
        
        # Should find 3 assignments: error_a, error_b, and None
        assert len(result.trigger_assignments) == 3
        
        values = [a.value for a in result.trigger_assignments]
        assert "error_a_trigger" in values
        assert "error_b_trigger" in values
        assert None in values  # The clearing assignment


class TestDictUpdatePatterns:
    """Tests for state.update() and |= dict merge patterns."""
    
    def test_detects_trigger_in_update_call(self):
        """state.update({"ask_user_trigger": ...}) should be detected."""
        code = '''
def func():
    state.update({"ask_user_trigger": "trigger_value", "pending_user_questions": ["q"]})
'''
        result = analyze_source(code, "test.py")
        
        assert len(result.trigger_assignments) == 1
        assert result.trigger_assignments[0].value == "trigger_value"
        assert result.trigger_assignments[0].in_dict_literal is True
        assert result.trigger_assignments[0].paired_in_dict is True
    
    def test_detects_unpaired_trigger_in_update_call(self):
        """Unpaired trigger in update() should be flagged."""
        code = '''
def func():
    state.update({"ask_user_trigger": "orphan_trigger"})
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 1
        assert errors[0].type == ViolationType.UNPAIRED_IN_DICT
    
    def test_detects_trigger_in_augmented_assign(self):
        """result |= {"ask_user_trigger": ...} should be detected."""
        code = '''
def func():
    result |= {"ask_user_trigger": "trigger_value", "pending_user_questions": ["q"]}
'''
        result = analyze_source(code, "test.py")
        
        assert len(result.trigger_assignments) == 1
        assert result.trigger_assignments[0].value == "trigger_value"
        assert result.trigger_assignments[0].paired_in_dict is True
    
    def test_detects_unpaired_trigger_in_augmented_assign(self):
        """Unpaired trigger in |= should be flagged."""
        code = '''
def func():
    result |= {"ask_user_trigger": "orphan_trigger"}
'''
        result = analyze_source(code, "test.py")
        violations = check_pairing_violations(result)
        
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 1
        assert errors[0].type == ViolationType.UNPAIRED_IN_DICT
    
    def test_detects_questions_in_update_call(self):
        """Questions in update() should be tracked."""
        code = '''
def func():
    state.update({"pending_user_questions": ["question"]})
'''
        result = analyze_source(code, "test.py")
        
        assert len(result.questions_assignments) == 1
        assert result.questions_assignments[0].is_empty_list is False
    
    def test_update_with_non_dict_arg_ignored(self):
        """update() with variable arg should be ignored (not a dict literal)."""
        code = '''
def func():
    data = {"ask_user_trigger": "x"}
    state.update(data)
'''
        result = analyze_source(code, "test.py")
        
        # Should detect the dict literal assignment, but not the update() call
        assert len(result.trigger_assignments) == 1  # From the dict literal
    
    def test_regular_function_update_ignored(self):
        """update() that's not a method call should be handled gracefully."""
        code = '''
def func():
    update({"ask_user_trigger": "x"})  # Regular function, not method
'''
        result = analyze_source(code, "test.py")
        
        # The dict inside is still a dict literal, so it should be detected
        assert len(result.trigger_assignments) == 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_handles_syntax_error(self, syntax_error_code):
        """Should handle syntax errors gracefully."""
        result = analyze_source(syntax_error_code, "test.py")
        
        assert len(result.parse_errors) > 0
        assert len(result.trigger_assignments) == 0
    
    def test_handles_empty_file(self, empty_file):
        """Should handle empty file."""
        result = analyze_source(empty_file, "test.py")
        
        assert len(result.trigger_assignments) == 0
        assert len(result.questions_lines) == 0
        assert len(result.parse_errors) == 0
    
    def test_handles_file_without_triggers(self, no_triggers):
        """Should handle file with no triggers."""
        result = analyze_source(no_triggers, "test.py")
        
        assert len(result.trigger_assignments) == 0
        assert len(result.questions_lines) == 0
        assert len(result.parse_errors) == 0
    
    def test_filepath_is_preserved(self, valid_dict_literal):
        """Should preserve filepath in result and assignments."""
        result = analyze_source(valid_dict_literal, "custom/path.py")
        
        assert result.filepath == "custom/path.py"
        assert result.trigger_assignments[0].filepath == "custom/path.py"


class TestQuestionsTracking:
    """Tests for tracking pending_user_questions assignments."""
    
    def test_tracks_questions_in_dict(self, valid_dict_literal):
        """Should track questions in dict literals."""
        result = analyze_source(valid_dict_literal, "test.py")
        
        assert len(result.questions_lines) == 1
    
    def test_tracks_questions_in_subscript(self, valid_subscript):
        """Should track questions in subscript assignments."""
        result = analyze_source(valid_subscript, "test.py")
        
        assert len(result.questions_lines) == 1
    
    def test_tracks_multiple_questions_assignments(self, multiple_triggers_same_function):
        """Should track all questions assignments."""
        result = analyze_source(multiple_triggers_same_function, "test.py")
        
        # Should find questions for error_a and error_b
        assert len(result.questions_lines) >= 2
