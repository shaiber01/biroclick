"""
Test trigger/questions pairing validation against the actual codebase.

This test ensures:
1. All ask_user_trigger assignments are paired with pending_user_questions
2. Dict literals contain both keys when setting a trigger
3. Subscript assignments have questions nearby

If this test fails, it means code is setting ask_user_trigger without
corresponding pending_user_questions, which would cause users to see
empty prompts.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validate_trigger_pairing import (
    validate_src_directory,
    ViolationType,
    format_violation,
)


class TestTriggerQuestionsPairing:
    """Tests that verify ask_user_trigger is always paired with pending_user_questions."""
    
    def test_no_unpaired_triggers_in_dict_literals(self):
        """
        CRITICAL: Dict literals with ask_user_trigger must include pending_user_questions.
        
        This is the most important check because dict literals are self-contained
        and there's no excuse for missing the pairing.
        """
        result = validate_src_directory()
        
        errors = [
            v for v in result.violations 
            if v.type == ViolationType.UNPAIRED_IN_DICT
        ]
        
        if errors:
            msg = (
                "\n\nFound ask_user_trigger in dict literals without pending_user_questions!\n"
                "This will cause users to see empty prompts.\n\n"
                "Violations:\n"
            )
            for v in errors:
                msg += f"  • {format_violation(v)}\n"
            msg += "\nFix: Add pending_user_questions alongside ask_user_trigger in the dict."
            pytest.fail(msg)
    
    def test_subscript_assignments_have_nearby_questions(self):
        """
        Subscript-style trigger assignments should have questions set within 10 lines.
        
        This catches patterns like:
            result["ask_user_trigger"] = "foo"
            # ... missing questions ...
        """
        result = validate_src_directory()
        
        errors = [
            v for v in result.violations 
            if v.type == ViolationType.UNPAIRED_SUBSCRIPT
        ]
        
        if errors:
            msg = (
                "\n\nFound ask_user_trigger subscript assignments without "
                "pending_user_questions within 10 lines:\n\n"
            )
            for v in errors:
                msg += f"  • {format_violation(v)}\n"
            msg += "\nFix: Add result[\"pending_user_questions\"] = [...] near the trigger assignment."
            pytest.fail(msg)
    
    def test_trigger_count_sanity_check(self):
        """
        Sanity check: questions count should be >= trigger count.
        
        Questions can exceed triggers because:
        - Questions are sometimes set without changing triggers
        - Dict literals containing both are counted at the same line
        
        But triggers should NOT significantly exceed questions.
        """
        result = validate_src_directory()
        
        # Print stats for visibility
        print(f"\n📊 Trigger/Questions balance:")
        print(f"   ask_user_trigger assignments (non-None): {result.trigger_count}")
        print(f"   pending_user_questions assignments: {result.questions_count}")
        
        # Questions can exceed triggers (OK), but triggers shouldn't exceed questions by much
        if result.trigger_count > result.questions_count:
            excess = result.trigger_count - result.questions_count
            tolerance = 5  # Allow small excess for edge cases
            assert excess <= tolerance, (
                f"Trigger count ({result.trigger_count}) exceeds questions count "
                f"({result.questions_count}) by {excess}. "
                f"This may indicate unpaired trigger assignments."
            )
        
        # Basic sanity: should have at least some of each
        assert result.trigger_count > 0, "Expected some trigger assignments"
        assert result.questions_count > 0, "Expected some questions assignments"
    
    def test_no_parse_errors(self):
        """Ensure all Python files in src/ can be parsed."""
        result = validate_src_directory()
        
        if result.parse_errors:
            msg = "\n\nFailed to parse some files:\n"
            for err in result.parse_errors:
                msg += f"  • {err}\n"
            pytest.fail(msg)
    
    def test_report_suspicious_variables_for_review(self):
        """
        INFO: Report variable-based trigger assignments for manual review.
        
        These might be intentional (preserving existing trigger) or bugs.
        This test prints warnings but doesn't fail.
        """
        result = validate_src_directory()
        
        warnings = [
            v for v in result.violations 
            if v.type == ViolationType.SUSPICIOUS_VARIABLE
        ]
        
        if warnings:
            print("\n\n📋 Variable-based trigger assignments (for manual review):")
            for v in warnings:
                print(f"  • {format_violation(v)}")
            print("\nThese may be intentional preservation of existing triggers.")
        
        # Don't fail, just report


class TestTriggerValidationStats:
    """Tests that verify the validation tool is working correctly on real code."""
    
    def test_finds_trigger_assignments(self):
        """The codebase should have some trigger assignments (sanity check)."""
        result = validate_src_directory()
        
        # We know the codebase has triggers, so this should be > 0
        assert result.trigger_count > 0, (
            "Expected to find trigger assignments in src/. "
            "Either the codebase changed dramatically or the tool is broken."
        )
    
    def test_finds_questions_assignments(self):
        """The codebase should have some questions assignments (sanity check)."""
        result = validate_src_directory()
        
        assert result.questions_count > 0, (
            "Expected to find pending_user_questions assignments in src/. "
            "Either the codebase changed dramatically or the tool is broken."
        )
    
    def test_analyzes_multiple_files(self):
        """Should analyze multiple files in src/."""
        result = validate_src_directory()
        
        assert result.files_analyzed > 10, (
            f"Expected to analyze many files, only found {result.files_analyzed}. "
            "Check that src/ directory exists and contains Python files."
        )
