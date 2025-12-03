"""Robustness and execution tests for `run_simulation`."""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.code_runner import get_platform_capabilities, run_simulation


class TestExecutionRobustness:
    """
    Real execution tests to verify robustness against crashes, resource limits,
    and bad output.
    """

    def test_binary_stdout_handling(self, tmp_path):
        """
        Test that simulation outputting invalid UTF-8 bytes to stdout doesn't
        crash the runner and is handled gracefully.
        """
        code = """
import sys
try:
    # Write binary data to stdout buffer to bypass text encoding
    sys.stdout.buffer.write(b'Binary junk: \\x80\\xff\\n')
    sys.stdout.flush()
except Exception as e:
    print(f"Write failed: {e}", file=sys.stderr)
"""
        result = run_simulation(
            code=code,
            stage_id="binary_test",
            output_dir=tmp_path,
        )

        assert result["exit_code"] == 0
        assert result["error"] is None

        if result["error"] and "UnicodeDecodeError" in str(result["error"]):
            pytest.fail(f"Code runner crashed on binary output: {result['error']}")

        assert "Binary junk" in result["stdout"] or "\ufffd" in result["stdout"]

    def test_real_timeout_enforcement(self, tmp_path):
        """Test that the timeout actually kills a sleeping process."""
        code = """
import time
import sys
print("Starting sleep", file=sys.stderr)
time.sleep(5)
print("Finished sleep", file=sys.stderr)
"""
        start = time.time()
        result = run_simulation(
            code=code,
            stage_id="timeout_test",
            output_dir=tmp_path,
            config={"timeout_seconds": 1},
        )
        duration = time.time() - start

        assert result["timeout_exceeded"] is True
        assert "timeout" in str(result["error"]).lower()
        assert duration < 4.0  # Should be closer to 1s than 5s

    def test_real_memory_limit_enforcement(self, tmp_path):
        """
        Test that memory limits are enforced on supported platforms.
        Attempts to allocate 200MB with a 50MB limit.
        """
        caps = get_platform_capabilities()
        if not caps["memory_limiting_available"]:
            pytest.skip("Memory limiting not available on this platform")

        import resource

        try:
            limit = 1024 * 1024 * 1024  # 1GB
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            if soft > limit:
                resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
                resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        except Exception as exc:
            pytest.skip(f"System rejects setrlimit (likely OS restriction): {exc}")

        code = """
import sys
import time

print("Allocating memory...", file=sys.stderr)
try:
    # Allocate ~200MB string
    # 200 * 1024 * 1024 bytes
    x = "a" * (200 * 1024 * 1024)
    print(f"Allocated {len(x)} bytes", file=sys.stderr)
    # Touch pages to ensure allocation happens
    y = x[::1024]
except MemoryError:
    print("Caught MemoryError in script", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Caught {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(2)
"""
        result = run_simulation(
            code=code,
            stage_id="mem_test",
            output_dir=tmp_path,
            config={"max_memory_gb": 0.05},
        )

        assert result["exit_code"] != 0
        assert (
            result["memory_exceeded"]
            or "kill" in str(result["error"]).lower()
            or "MemoryError" in result["stderr"]
        )

    def test_large_output_handling(self, tmp_path):
        """Test handling of large stdout to ensure no hangs."""
        code = """
for i in range(10000):
    print(f"Line {i} of spam output to test buffer handling")
"""
        result = run_simulation(
            code=code,
            stage_id="spam_test",
            output_dir=tmp_path,
        )

        assert result["exit_code"] == 0
        assert len(result["stdout"]) > 10000
        assert "Line 9999" in result["stdout"]

    def test_syntax_error_reporting(self, tmp_path):
        """Test that syntax errors are correctly reported."""
        code = "this is not valid python"
        result = run_simulation(code=code, stage_id="syntax", output_dir=tmp_path)

        assert result["exit_code"] != 0
        assert "SyntaxError" in result["stderr"]

    def test_run_simulation_detects_signal_kill(self, tmp_path):
        """Test detection of process killed by signal (e.g. OOM killer)."""
        with patch("src.code_runner.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = -9  # SIGKILL
            mock_result.stdout = ""
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = run_simulation(
                code="print('running')",
                stage_id="kill_test",
                output_dir=tmp_path,
            )

            error_str = str(result["error"]).lower()
            assert "killed" in error_str or "signal" in error_str

