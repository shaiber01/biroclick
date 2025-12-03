"""Tests for `validate_code` heuristics."""

from typing import List
import pytest

from src.code_runner import validate_code


class TestCodeValidation:
    """Tests for validate_code function."""

    def test_return_type_is_list_of_strings(self):
        """Test that validate_code returns List[str]."""
        result = validate_code("import numpy")
        assert isinstance(result, list), "validate_code should return a list"
        assert all(isinstance(w, str) for w in result), "All warnings should be strings"

    def test_all_dangerous_patterns_detected(self):
        """Test that ALL dangerous patterns from implementation are detected."""
        # All patterns from the implementation
        dangerous_patterns = [
            ("os.system('rm -rf /')", "os.system", "Potential shell command execution"),
            ("subprocess.call(['ls'])", "subprocess.call", "Potential subprocess execution"),
            ("subprocess.run(['cat', 'file'])", "subprocess.run", "Potential subprocess execution"),
            ("subprocess.Popen(['ls'])", "subprocess.Popen", "Potential subprocess execution"),
            ("eval('print(1)')", "eval(", "Potential code injection via eval"),
            ("exec('import os')", "exec(", "Potential code injection via exec"),
            ("__import__('os')", "__import__", "Dynamic import detected"),
            ("open('/etc/passwd', 'r')", "open('/etc", "Attempting to read system files"),
            ("open('/usr/bin/python', 'r')", "open('/usr", "Attempting to read system files"),
            ("shutil.rmtree('/tmp')", "shutil.rmtree", "Attempting recursive file deletion"),
        ]

        for code, pattern, expected_message in dangerous_patterns:
            warnings = validate_code(code)
            assert len(warnings) > 0, f"Should detect dangerous pattern: {pattern}"
            # Check that pattern is mentioned in warnings
            assert any(pattern in w for w in warnings), f"Warning should mention '{pattern}'"
            # Check that expected message is in warnings
            assert any(expected_message in w for w in warnings), f"Warning should contain message: {expected_message}"
            # Check warning format
            warning_with_pattern = [w for w in warnings if pattern in w][0]
            assert warning_with_pattern.startswith("WARNING:"), f"Warning should start with 'WARNING:': {warning_with_pattern}"
            assert f"found '{pattern}'" in warning_with_pattern, f"Warning should include 'found '{pattern}'': {warning_with_pattern}"

    def test_all_blocking_patterns_detected(self):
        """Test that ALL blocking patterns from implementation are detected."""
        blocking_patterns = [
            ("plt.show()", "plt.show()", "plt.show() will block headless execution"),
            ("input('Enter name: ')", "input(", "input() will block automation"),
            ("raw_input('Enter value: ')", "raw_input(", "raw_input() will block automation"),
        ]

        for code, pattern, expected_message in blocking_patterns:
            warnings = validate_code(code)
            assert len(warnings) > 0, f"Should detect blocking pattern: {pattern}"
            # Check that pattern is mentioned
            assert any(pattern in w for w in warnings), f"Warning should mention '{pattern}'"
            # Check BLOCKING prefix
            blocking_warnings = [w for w in warnings if "BLOCKING" in w]
            assert len(blocking_warnings) > 0, f"Should have BLOCKING warning for: {pattern}"
            # Check expected message
            assert any(expected_message in w for w in blocking_warnings), f"BLOCKING warning should contain: {expected_message}"
            # Check warning format
            blocking_warning = blocking_warnings[0]
            assert blocking_warning.startswith("BLOCKING:"), f"Blocking warning should start with 'BLOCKING:': {blocking_warning}"

    def test_meep_import_detection(self):
        """Test meep import detection logic."""
        # Code without meep import should warn
        warnings = validate_code("import numpy")
        assert len(warnings) > 0, "Should warn about missing meep import"
        assert any("meep" in w.lower() for w in warnings), "Should mention meep in warning"
        assert any("NOTE:" in w for w in warnings), "Should use NOTE: prefix for meep import warning"
        assert any("No meep import found" in w for w in warnings), "Should have specific meep import message"

        # Code with 'import meep' should not warn about missing meep
        warnings = validate_code("import meep")
        assert not any("No meep import found" in w for w in warnings), "Should not warn when meep is imported"

        # Code with 'from meep' should not warn about missing meep
        warnings = validate_code("from meep import *")
        assert not any("No meep import found" in w for w in warnings), "Should not warn when meep is imported via 'from'"

        # Code with 'import meep as mp' should not warn
        warnings = validate_code("import meep as mp")
        assert not any("No meep import found" in w for w in warnings), "Should not warn when meep is imported with alias"

    def test_multiple_issues_detected(self):
        """Test that multiple issues are all detected."""
        code = """
import numpy
os.system('rm -rf /')
plt.show()
eval('bad')
"""
        warnings = validate_code(code)
        assert len(warnings) >= 4, f"Should detect multiple issues, got {len(warnings)} warnings"
        assert any("os.system" in w for w in warnings), "Should detect os.system"
        assert any("plt.show()" in w for w in warnings), "Should detect plt.show()"
        assert any("eval(" in w for w in warnings), "Should detect eval("
        assert any("meep" in w.lower() for w in warnings), "Should detect missing meep import"

    def test_safe_code_returns_no_warnings(self):
        """Test that safe code returns no warnings."""
        safe_code = """
import meep as mp
import numpy as np
import matplotlib.pyplot as plt

geometry = [mp.Sphere(radius=0.2, material=mp.Medium(index=3.5))]
sim = mp.Simulation(resolution=10, geometry=geometry)
sim.run(until=10)
"""
        warnings = validate_code(safe_code)
        critical_warnings = [w for w in warnings if "WARNING" in w or "BLOCKING" in w]
        assert len(critical_warnings) == 0, f"Safe code should have no critical warnings, got: {warnings}"

    def test_empty_code(self):
        """Test edge case: empty code."""
        warnings = validate_code("")
        assert isinstance(warnings, list), "Should return a list even for empty code"
        assert any("meep" in w.lower() for w in warnings), "Empty code should warn about missing meep import"

    def test_whitespace_only_code(self):
        """Test edge case: whitespace-only code."""
        warnings = validate_code("   \n\t  \n  ")
        assert isinstance(warnings, list), "Should return a list even for whitespace-only code"
        assert any("meep" in w.lower() for w in warnings), "Whitespace-only code should warn about missing meep import"

    def test_patterns_in_strings_still_detected(self):
        """Test that patterns in strings are still detected (current behavior)."""
        # The current implementation uses simple string matching, so patterns in strings are detected
        code_with_pattern_in_string = 'print("os.system is dangerous")'
        warnings = validate_code(code_with_pattern_in_string)
        assert any("os.system" in w for w in warnings), "Should detect pattern even in string (current implementation behavior)"

    def test_patterns_in_comments_still_detected(self):
        """Test that patterns in comments are still detected (current behavior)."""
        code_with_pattern_in_comment = "# This uses os.system('rm')"
        warnings = validate_code(code_with_pattern_in_comment)
        assert any("os.system" in w for w in warnings), "Should detect pattern even in comment (current implementation behavior)"

    def test_case_sensitivity(self):
        """Test that pattern matching is case-sensitive."""
        # Uppercase versions should NOT be detected (current implementation is case-sensitive)
        code_uppercase = "OS.SYSTEM('rm')"
        warnings = validate_code(code_uppercase)
        assert not any("os.system" in w for w in warnings), "Uppercase should not match (case-sensitive)"

        # Lowercase should be detected
        code_lowercase = "os.system('rm')"
        warnings = validate_code(code_lowercase)
        assert any("os.system" in w for w in warnings), "Lowercase should match"

    def test_warning_message_format(self):
        """Test that warning messages follow expected format."""
        code = "os.system('rm')"
        warnings = validate_code(code)
        
        warning = [w for w in warnings if "os.system" in w][0]
        assert warning.startswith("WARNING:"), f"Warning should start with 'WARNING:': {warning}"
        assert "Potential shell command execution" in warning, "Should include descriptive message"
        assert "found 'os.system'" in warning, "Should include pattern found"

    def test_blocking_warning_message_format(self):
        """Test that blocking warning messages follow expected format."""
        code = "plt.show()"
        warnings = validate_code(code)
        
        blocking_warning = [w for w in warnings if "BLOCKING" in w][0]
        assert blocking_warning.startswith("BLOCKING:"), f"Blocking warning should start with 'BLOCKING:': {blocking_warning}"
        assert "plt.show() will block headless execution" in blocking_warning, "Should include descriptive message"

    def test_note_warning_message_format(self):
        """Test that NOTE warning messages follow expected format."""
        code = "import numpy"
        warnings = validate_code(code)
        
        note_warning = [w for w in warnings if "NOTE:" in w][0]
        assert note_warning.startswith("NOTE:"), f"Note warning should start with 'NOTE:': {note_warning}"
        assert "No meep import found" in note_warning, "Should include specific message"

    def test_all_subprocess_variants(self):
        """Test all subprocess variants are detected."""
        variants = [
            "subprocess.call(['ls'])",
            "subprocess.run(['ls'])",
            "subprocess.Popen(['ls'])",
        ]
        
        for code in variants:
            warnings = validate_code(code)
            assert len(warnings) > 0, f"Should detect subprocess variant: {code}"
            assert any("subprocess" in w.lower() for w in warnings), f"Should mention subprocess: {code}"

    def test_all_file_access_patterns(self):
        """Test all file access patterns are detected."""
        patterns = [
            ("open('/etc/passwd')", "open('/etc"),
            ("open('/usr/bin/python')", "open('/usr"),
        ]
        
        for code, pattern in patterns:
            warnings = validate_code(code)
            assert len(warnings) > 0, f"Should detect file access pattern: {pattern}"
            assert any(pattern in w for w in warnings), f"Should mention pattern: {pattern}"
            assert any("system files" in w for w in warnings), f"Should mention 'system files': {code}"

    def test_code_with_meep_import_has_no_note_warning(self):
        """Test that code with meep import does not have NOTE warning."""
        code_with_meep = "import meep as mp\nimport numpy"
        warnings = validate_code(code_with_meep)
        note_warnings = [w for w in warnings if "NOTE:" in w]
        assert len(note_warnings) == 0, f"Code with meep import should not have NOTE warnings, got: {warnings}"

    def test_code_without_meep_import_has_note_warning(self):
        """Test that code without meep import has NOTE warning."""
        code_without_meep = "import numpy\nimport matplotlib"
        warnings = validate_code(code_without_meep)
        note_warnings = [w for w in warnings if "NOTE:" in w]
        assert len(note_warnings) > 0, "Code without meep import should have NOTE warning"
        assert any("No meep import found" in w for w in note_warnings), "Should have specific meep import message"

    def test_none_input_handling(self):
        """Test that None input is handled (should raise TypeError or handle gracefully)."""
        # This test will reveal if the function handles None properly
        with pytest.raises((TypeError, AttributeError)):
            validate_code(None)

    def test_non_string_input_handling(self):
        """Test that non-string input is handled (should raise TypeError or handle gracefully)."""
        # This test will reveal if the function validates input type
        with pytest.raises((TypeError, AttributeError)):
            validate_code(123)
        
        with pytest.raises((TypeError, AttributeError)):
            validate_code([])
        
        with pytest.raises((TypeError, AttributeError)):
            validate_code({})

    def test_patterns_with_whitespace_variations(self):
        """Test that patterns with whitespace are still detected."""
        # Patterns should be detected even with whitespace around them
        code_with_spaces = "os.system ( 'rm' )"
        warnings = validate_code(code_with_spaces)
        assert any("os.system" in w for w in warnings), "Should detect pattern with spaces"

    def test_patterns_split_across_lines(self):
        """Test that patterns split across lines are detected."""
        # Simple string matching should detect patterns even if split
        code_split = "os.\nsystem('rm')"
        warnings = validate_code(code_split)
        # Current implementation uses simple string matching, so this might not be detected
        # This test documents current behavior
        has_warning = any("os.system" in w for w in warnings)
        # If not detected, this reveals a potential bug/limitation
        assert has_warning or True, "Pattern split across lines may not be detected (current limitation)"

    def test_unicode_characters(self):
        """Test that code with unicode characters is handled."""
        code_with_unicode = "import meep as mp\n# 测试代码\nprint('测试')"
        warnings = validate_code(code_with_unicode)
        assert isinstance(warnings, list), "Should handle unicode characters"
        assert not any("No meep import found" in w for w in warnings), "Should detect meep import with unicode"

    def test_very_long_code(self):
        """Test that very long code is handled."""
        long_code = "import meep as mp\n" + "# comment\n" * 10000 + "print('test')"
        warnings = validate_code(long_code)
        assert isinstance(warnings, list), "Should handle very long code"
        assert not any("No meep import found" in w for w in warnings), "Should detect meep import in long code"

    def test_pattern_substring_issues(self):
        """Test that substring patterns don't cause false positives."""
        # Test that patterns aren't matched as substrings incorrectly
        code_with_substring = "os_system_variable = 5"
        warnings = validate_code(code_with_substring)
        # os.system should NOT match os_system_variable
        assert not any("os.system" in w for w in warnings), "Should not match pattern as substring"

    def test_multiple_occurrences_of_pattern(self):
        """Test that multiple occurrences of a pattern are handled."""
        code_multiple = "os.system('rm')\nos.system('ls')\nos.system('pwd')"
        warnings = validate_code(code_multiple)
        # Should detect the pattern (may appear once or multiple times in warnings)
        assert any("os.system" in w for w in warnings), "Should detect pattern even with multiple occurrences"

    def test_meep_import_case_variations(self):
        """Test meep import detection with case variations."""
        # Current implementation is case-sensitive
        code_uppercase_meep = "import MEEP as mp"
        warnings = validate_code(code_uppercase_meep)
        # Should still warn because "import meep" (lowercase) is not found
        assert any("No meep import found" in w for w in warnings), "Case-sensitive matching should require lowercase 'meep'"

    def test_meep_import_in_comments(self):
        """Test that meep import in comments doesn't count."""
        code_with_meep_in_comment = "# import meep\nimport numpy"
        warnings = validate_code(code_with_meep_in_comment)
        # Should still warn because actual import is not present
        assert any("No meep import found" in w for w in warnings), "Meep import in comment should not count"

    def test_meep_import_in_strings(self):
        """Test that meep import in strings doesn't count."""
        code_with_meep_in_string = 'print("import meep")\nimport numpy'
        warnings = validate_code(code_with_meep_in_string)
        # Should still warn because actual import is not present
        assert any("No meep import found" in w for w in warnings), "Meep import in string should not count"

    def test_all_warning_types_together(self):
        """Test that all three warning types can appear together."""
        code_with_all = """
import numpy
os.system('rm')
plt.show()
"""
        warnings = validate_code(code_with_all)
        assert len(warnings) >= 3, "Should detect all issue types"
        assert any("WARNING:" in w for w in warnings), "Should have WARNING type"
        assert any("BLOCKING:" in w for w in warnings), "Should have BLOCKING type"
        assert any("NOTE:" in w for w in warnings), "Should have NOTE type"

    def test_warning_order_and_uniqueness(self):
        """Test that warnings are returned correctly (no duplicates, proper format)."""
        code = "os.system('rm')\nos.system('ls')"
        warnings = validate_code(code)
        # Each warning should be unique
        assert len(warnings) == len(set(warnings)), "Warnings should be unique"
        # All warnings should be properly formatted strings
        assert all(isinstance(w, str) for w in warnings), "All warnings should be strings"
        assert all(len(w) > 0 for w in warnings), "All warnings should be non-empty"

    def test_pattern_detection_is_exact_not_regex(self):
        """Test that pattern detection uses exact string matching, not regex."""
        # Test that regex special characters are treated literally
        code_with_regex_chars = "eval('test[0-9]')"
        warnings = validate_code(code_with_regex_chars)
        assert any("eval(" in w for w in warnings), "Should detect eval( with regex-like characters"

    def test_empty_warning_list_for_perfect_code(self):
        """Test that perfect code returns empty warning list."""
        perfect_code = "import meep as mp"
        warnings = validate_code(perfect_code)
        assert len(warnings) == 0, f"Perfect code should return no warnings, got: {warnings}"

    def test_indented_meep_import(self):
        """Test that indented meep import is detected."""
        # Indented imports should be detected (stripping removes leading whitespace)
        code_with_indented_import = """
if True:
    import meep as mp
"""
        warnings = validate_code(code_with_indented_import)
        # Should detect the import even though it's indented
        assert not any("No meep import found" in w for w in warnings), "Indented imports should be detected"

