"""
Tests for the Code Runner Module.

Tests platform detection, code validation, runtime estimation, and the
error result helper function.
"""

import os
import pytest
import shutil
import time
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.code_runner import (
    detect_platform,
    get_platform_capabilities,
    validate_code,
    estimate_runtime,
    _make_error_result,
    run_simulation,
    run_code_node,
    PlatformCapabilities,
    ExecutionResult,
    _list_output_files,
)


# ═══════════════════════════════════════════════════════════════════════
# Platform Detection Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPlatformDetection:
    """Tests for platform detection functions."""
    
    def test_detect_platform_returns_valid_capabilities(self):
        """Test that detect_platform returns a valid capabilities dict."""
        caps = detect_platform()
        
        assert isinstance(caps, dict)
        assert "platform" in caps
        assert "memory_limiting_available" in caps
        assert "process_group_kill_available" in caps
        assert "preexec_fn_available" in caps
        assert "is_wsl" in caps
        assert "warnings" in caps
        assert isinstance(caps["warnings"], list)
    
    def test_detect_platform_identifies_current_os(self):
        """Test that platform detection matches sys.platform."""
        caps = detect_platform()
        
        if sys.platform == "darwin":
            assert caps["platform"] == "macos"
        elif sys.platform == "win32":
            assert caps["platform"] == "windows"
        elif sys.platform.startswith("linux"):
            # WSL check might override "linux" to "wsl"
            if caps["is_wsl"]:
                assert caps["platform"] == "wsl"
            else:
                assert caps["platform"] == "linux"

    def test_get_platform_capabilities_cached(self):
        """Test that get_platform_capabilities returns cached result."""
        caps1 = get_platform_capabilities()
        caps2 = get_platform_capabilities()
        assert caps1 is caps2
    

# ═══════════════════════════════════════════════════════════════════════
# Code Validation Tests
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# Runtime Estimation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRuntimeEstimation:
    """Tests for estimate_runtime function."""
    
    def test_estimate_runtime_heuristics(self):
        """Test that heuristics increase runtime estimate."""
        base = estimate_runtime("import meep")
        
        code_3d = "mp.Vector3(1,1,1)\n" * 3
        est_3d = estimate_runtime(code_3d)
        assert est_3d["estimated_minutes"] > base["estimated_minutes"]
        assert est_3d["features_detected"]["is_3d"]
        
        code_sweep = "for i in range(10): pass"
        est_sweep = estimate_runtime(code_sweep)
        assert est_sweep["estimated_minutes"] > base["estimated_minutes"]
        assert est_sweep["features_detected"]["has_sweep"]

    def test_estimate_runtime_timeout_buffer(self):
        """Test timeout is buffered."""
        est = estimate_runtime("import meep")
        assert est["recommended_timeout_seconds"] >= est["estimated_minutes"] * 60 * 2


# ═══════════════════════════════════════════════════════════════════════
# Robustness & Real Execution Tests
# ═══════════════════════════════════════════════════════════════════════

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
        # Python code that writes non-UTF8 bytes to stdout
        # \x80 is not a valid start byte in UTF-8
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
            output_dir=tmp_path
        )
        
        assert result["exit_code"] == 0
        assert result["error"] is None
        # Should contain the replacement characters or handled text
        # If strict decoding was used, this test would likely have failed with uncaught exception
        # or empty stdout if subprocess swallowed it.
        # We expect robust handling (e.g. replacement chars or ignored errors).
        # Wait - run_simulation uses text=True in subprocess.run.
        # If the output is invalid utf-8, subprocess.run with text=True MIGHT crash unless errors='replace'
        # Let's see if it crashes or handles it. If it crashes, we need to fix code_runner.
        # For now, let's assert we got something back.
        
        # Note: If it crashes inside code_runner, result['error'] will be set by the catch-all block.
        if result['error'] and "UnicodeDecodeError" in str(result['error']):
             # This confirms we need to fix code_runner to use errors='replace'
             pytest.fail(f"Code runner crashed on binary output: {result['error']}")
             
        # If no crash, check if we got the text part
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
            config={"timeout_seconds": 1}
        )
        duration = time.time() - start
        
        assert result["timeout_exceeded"] is True
        assert "timeout" in str(result["error"]).lower()
        assert duration < 4.0 # Should be closer to 1s than 5s
        
    def test_real_memory_limit_enforcement(self, tmp_path):
        """
        Test that memory limits are enforced on supported platforms.
        Attempts to allocate 200MB with a 50MB limit.
        """
        caps = get_platform_capabilities()
        if not caps["memory_limiting_available"]:
            pytest.skip("Memory limiting not available on this platform")

        # Pre-check if we can actually set limits on this specific environment
        import resource
        try:
            limit = 1024 * 1024 * 1024 # 1GB
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            # Only try if we aren't already lower than 1GB (unlikely)
            if soft > limit:
                resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
                # Reset (best effort)
                resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        except Exception as e:
             pytest.skip(f"System rejects setrlimit (likely OS restriction): {e}")

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
        # Set limit to 50MB (0.05 GB)
        result = run_simulation(
            code=code,
            stage_id="mem_test",
            output_dir=tmp_path,
            config={"max_memory_gb": 0.05}
        )
        
        # The process should fail either with MemoryError (caught inside) 
        # or be killed by OOM killer
        assert result["exit_code"] != 0
        assert result["memory_exceeded"] or "kill" in str(result["error"]).lower() or "MemoryError" in result["stderr"]

    def test_large_output_handling(self, tmp_path):
        """Test handling of large stdout to ensure no hangs."""
        code = """
for i in range(10000):
    print(f"Line {i} of spam output to test buffer handling")
"""
        result = run_simulation(
            code=code,
            stage_id="spam_test",
            output_dir=tmp_path
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


# ═══════════════════════════════════════════════════════════════════════
# Integration & Helper Tests
# ═══════════════════════════════════════════════════════════════════════

class TestHelpers:
    """Tests for helper functions."""
    
    def test_make_error_result(self):
        """Test error result construction."""
        res = _make_error_result("oops", exit_code=5)
        assert res["error"] == "oops"
        assert res["exit_code"] == 5
        assert res["output_files"] == []

    def test_list_output_files(self, tmp_path):
        """Test listing files."""
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        (tmp_path / "ignore.py").touch()
        
        files = _list_output_files(tmp_path, exclude=["ignore.py"])
        assert "a.txt" in files
        assert "b.txt" in files
        assert "ignore.py" not in files

class TestRunCodeNode:
    """Tests for run_code_node integration."""
    
    @patch("src.code_runner.run_simulation")
    def test_run_code_node_extracts_config(self, mock_run):
        """Test that config is extracted correctly from state."""
        mock_run.return_value = {
            "stdout": "", "stderr": "", "exit_code": 0, 
            "output_files": [], "runtime_seconds": 1.0, 
            "error": None, "memory_exceeded": False, "timeout_exceeded": False
        }
        
        state = {
            "code": "import meep",
            "current_stage_id": "stage1",
            "paper_id": "paper1",
            "plan": {
                "stages": [
                    {"stage_id": "stage1", "runtime_budget_minutes": 10}
                ]
            },
            "runtime_config": {
                "max_memory_gb": 16.0,
                "max_cpu_cores": 8
            }
        }
        
        run_code_node(state)
        
        # Check call args
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        kwargs = call_args.kwargs
        
        assert kwargs["code"] == "import meep"
        assert kwargs["stage_id"] == "stage1"
        assert kwargs["config"]["timeout_seconds"] == 600  # 10 * 60
        assert kwargs["config"]["max_memory_gb"] == 16.0
        assert kwargs["config"]["max_cpu_cores"] == 8

    def test_run_code_node_missing_code(self):
        """Test that missing code returns error."""
        state = {"current_stage_id": "test"} # no 'code' key
        result = run_code_node(state)
        assert "No simulation code provided" in result["run_error"]
