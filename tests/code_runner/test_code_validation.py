"""Tests for `validate_code` heuristics."""

from src.code_runner import validate_code


class TestCodeValidation:
    """Tests for validate_code function."""

    def test_validate_code_detects_dangerous_patterns(self):
        """Test that dangerous patterns are flagged."""
        dangerous = [
            ("os.system('rm')", "os.system"),
            ("subprocess.call(['ls'])", "subprocess.call"),
            ("eval('print(1)')", "eval("),
            ("exec('import os')", "exec("),
            ("__import__('os')", "__import__"),
            ("open('/etc/passwd')", "open('/etc"),
        ]

        for code, pattern in dangerous:
            warnings = validate_code(code)
            assert any(pattern in w for w in warnings), f"Failed to detect {pattern}"

    def test_validate_code_detects_blocking_calls(self):
        """Test that blocking calls are flagged."""
        blocking = [
            ("plt.show()", "plt.show()"),
            ("input('name')", "input("),
        ]

        for code, pattern in blocking:
            warnings = validate_code(code)
            assert any(pattern in w for w in warnings)
            assert any("BLOCKING" in w for w in warnings)

    def test_validate_code_checks_imports(self):
        """Test missing meep import warning."""
        warnings = validate_code("import numpy")
        assert any("meep" in w.lower() for w in warnings)

        warnings = validate_code("import meep as mp")
        critical = [w for w in warnings if "WARNING" in w or "BLOCKING" in w]
        assert len(critical) == 0

