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
    validate_file,
    analyze_file,
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
        
        # Verify we actually checked something (not just an empty scan)
        assert result.files_analyzed > 0, "No files were analyzed"
    
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
        
        # Verify we actually checked something
        assert result.files_analyzed > 0, "No files were analyzed"
    
    def test_trigger_count_sanity_check(self):
        """
        Strict sanity check for trigger/questions balance.
        
        Expected behavior based on codebase analysis:
        - trigger_handlers.py has ~25 question-only lines (clarification handlers)
        - This means questions_count > trigger_count is EXPECTED
        - But trigger_count > questions_count would indicate BUGS
        
        This test will FAIL if:
        - Triggers exceed questions (indicates unpaired triggers)
        - Counts change dramatically (indicates structural changes needing review)
        """
        result = validate_src_directory()
        
        # Print stats for debugging
        print(f"\n📊 Trigger/Questions balance:")
        print(f"   ask_user_trigger assignments (non-None): {result.trigger_count}")
        print(f"   pending_user_questions assignments: {result.questions_count}")
        print(f"   Difference (questions - triggers): {result.questions_count - result.trigger_count}")
        
        # STRICT CHECK 1: Triggers must NEVER exceed questions
        # If this fails, there are unpaired trigger assignments
        assert result.trigger_count <= result.questions_count, (
            f"CRITICAL: Trigger count ({result.trigger_count}) exceeds questions count "
            f"({result.questions_count}). This indicates unpaired trigger assignments "
            f"that will cause users to see empty prompts!"
        )
        
        # STRICT CHECK 2: Verify counts are in expected ranges
        # These ranges are based on current codebase analysis (Dec 2024)
        # If they change significantly, it indicates structural changes needing review
        MIN_EXPECTED_TRIGGERS = 25  # Currently ~32
        MAX_EXPECTED_TRIGGERS = 50
        MIN_EXPECTED_QUESTIONS = 50  # Currently ~59
        MAX_EXPECTED_QUESTIONS = 80
        
        assert result.trigger_count >= MIN_EXPECTED_TRIGGERS, (
            f"Trigger count ({result.trigger_count}) is below expected minimum ({MIN_EXPECTED_TRIGGERS}). "
            f"This could indicate the tool is missing assignments or the codebase changed significantly."
        )
        assert result.trigger_count <= MAX_EXPECTED_TRIGGERS, (
            f"Trigger count ({result.trigger_count}) exceeds expected maximum ({MAX_EXPECTED_TRIGGERS}). "
            f"Review new trigger assignments to ensure they're all properly paired."
        )
        assert result.questions_count >= MIN_EXPECTED_QUESTIONS, (
            f"Questions count ({result.questions_count}) is below expected minimum ({MIN_EXPECTED_QUESTIONS}). "
            f"This could indicate the tool is missing assignments or the codebase changed significantly."
        )
        assert result.questions_count <= MAX_EXPECTED_QUESTIONS, (
            f"Questions count ({result.questions_count}) exceeds expected maximum ({MAX_EXPECTED_QUESTIONS}). "
            f"This is likely OK but should be reviewed."
        )
    
    def test_no_parse_errors(self):
        """Ensure all Python files in src/ can be parsed."""
        result = validate_src_directory()
        
        assert len(result.parse_errors) == 0, (
            f"Failed to parse {len(result.parse_errors)} file(s):\n" +
            "\n".join(f"  • {err}" for err in result.parse_errors)
        )
    
    def test_suspicious_variables_are_known_patterns(self):
        """
        Variable-based trigger assignments must be known preservation patterns.
        
        This test FAILS if there are suspicious variables that aren't in the
        allowed list. Unknown variables could be bugs.
        """
        result = validate_src_directory()
        
        warnings = [
            v for v in result.violations 
            if v.type == ViolationType.SUSPICIOUS_VARIABLE
        ]
        
        # Currently allowed: supervisor.py preserves 'ask_user_trigger' variable
        # If new suspicious variables appear, they must be reviewed and either:
        # 1. Added to the PRESERVATION_VARS list in the tool
        # 2. Fixed in the codebase
        MAX_ALLOWED_SUSPICIOUS = 0  # We expect all variables to be known patterns
        
        if len(warnings) > MAX_ALLOWED_SUSPICIOUS:
            msg = (
                f"\n\nFound {len(warnings)} suspicious variable-based trigger assignments "
                f"(max allowed: {MAX_ALLOWED_SUSPICIOUS}):\n\n"
            )
            for v in warnings:
                msg += f"  • {format_violation(v)}\n"
            msg += (
                "\nThese could be bugs. Either:\n"
                "1. Add the variable name to PRESERVATION_VARS in validate_trigger_pairing.py\n"
                "2. Fix the code to use a string literal with proper pairing"
            )
            pytest.fail(msg)


class TestTriggerValidationStats:
    """Tests that verify the validation tool is working correctly on real code."""
    
    def test_finds_expected_trigger_files(self):
        """
        Verify the tool finds triggers in the expected files.
        
        This catches regressions where the tool stops detecting certain patterns.
        """
        expected_files_with_triggers = [
            "src/agents/execution.py",
            "src/agents/planning.py",
            "src/agents/code.py",
            "src/agents/design.py",
            "src/agents/analysis.py",
            "src/agents/supervision/supervisor.py",
            "src/agents/user_interaction.py",
        ]
        
        src_dir = PROJECT_ROOT / "src"
        found_triggers_in = set()
        
        for filepath in src_dir.rglob("*.py"):
            analysis = analyze_file(filepath)
            if analysis.trigger_assignments:
                # Store relative path for comparison
                rel_path = str(filepath.relative_to(PROJECT_ROOT))
                found_triggers_in.add(rel_path)
        
        for expected in expected_files_with_triggers:
            assert expected in found_triggers_in, (
                f"Expected to find triggers in {expected} but didn't. "
                f"Either the file changed or the tool has a detection bug."
            )
    
    def test_trigger_handlers_has_only_questions(self):
        """
        Verify trigger_handlers.py has questions but no triggers.
        
        This is expected behavior: handlers provide clarification questions
        for existing triggers, they don't set new ones.
        """
        handlers_path = PROJECT_ROOT / "src" / "agents" / "supervision" / "trigger_handlers.py"
        
        if handlers_path.exists():
            analysis = analyze_file(handlers_path)
            
            # Should have many questions (clarification handlers)
            assert len(analysis.questions_lines) >= 20, (
                f"Expected trigger_handlers.py to have many question lines (clarifications), "
                f"but only found {len(analysis.questions_lines)}. "
                f"This could indicate the tool isn't detecting questions properly."
            )
            
            # Should have NO trigger assignments
            non_none_triggers = [a for a in analysis.trigger_assignments if a.value_type != "none"]
            assert len(non_none_triggers) == 0, (
                f"trigger_handlers.py should NOT set new triggers (only provide questions), "
                f"but found {len(non_none_triggers)} trigger assignments. "
                f"This is a codebase design violation."
            )
    
    def test_analyzes_all_agent_files(self):
        """
        Verify all Python files in src/agents/ are analyzed.
        
        This catches cases where files are skipped due to encoding or other issues.
        """
        src_dir = PROJECT_ROOT / "src"
        agents_dir = src_dir / "agents"
        
        if not agents_dir.exists():
            pytest.skip("agents directory not found")
        
        expected_py_files = list(agents_dir.rglob("*.py"))
        # Exclude __init__.py and __pycache__
        expected_py_files = [
            f for f in expected_py_files 
            if f.name != "__init__.py" and "__pycache__" not in str(f)
        ]
        
        result = validate_src_directory()
        
        # We should analyze at least as many files as there are in agents/
        assert result.files_analyzed >= len(expected_py_files), (
            f"Expected to analyze at least {len(expected_py_files)} agent files, "
            f"but only analyzed {result.files_analyzed} total files."
        )
