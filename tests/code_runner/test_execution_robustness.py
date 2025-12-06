"""Robustness and execution tests for `run_simulation`."""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.code_runner import (
    estimate_runtime,
    get_platform_capabilities,
    run_simulation,
    validate_code,
    run_code_node,
)


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

        # Verify all required fields are present
        assert "exit_code" in result
        assert "stdout" in result
        assert "stderr" in result
        assert "error" in result
        assert "output_files" in result
        assert "runtime_seconds" in result
        assert "memory_exceeded" in result
        assert "timeout_exceeded" in result

        # Verify execution succeeded
        assert result["exit_code"] == 0, f"Expected exit code 0, got {result['exit_code']}"
        assert result["error"] is None, f"Expected no error, got: {result['error']}"
        assert result["timeout_exceeded"] is False
        assert result["memory_exceeded"] is False

        # Verify binary data was handled (either decoded or replaced with replacement char)
        assert "Binary junk" in result["stdout"] or "\ufffd" in result["stdout"], \
            f"Binary data not found in stdout: {result['stdout'][:100]}"

        # Verify runtime is reasonable (should be < 1 second for this simple code)
        assert result["runtime_seconds"] > 0, "Runtime should be positive"
        assert result["runtime_seconds"] < 5.0, f"Runtime too long: {result['runtime_seconds']}s"

        # Verify script file was created
        script_path = tmp_path / "simulation_binary_test.py"
        assert script_path.exists(), "Script file should exist"

    def test_real_timeout_enforcement(self, tmp_path):
        """Test that the timeout actually kills a sleeping process."""
        code = """
import time
import sys
print("Starting sleep", file=sys.stderr)
time.sleep(2)
print("Finished sleep", file=sys.stderr)
"""
        start = time.time()
        result = run_simulation(
            code=code,
            stage_id="timeout_test",
            output_dir=tmp_path,
            config={"timeout_seconds": 0.2},
        )
        duration = time.time() - start

        # Verify timeout was detected
        assert result["timeout_exceeded"] is True, "Timeout should be marked as exceeded"
        assert result["exit_code"] != 0, "Timeout should result in non-zero exit code"
        assert result["error"] is not None, "Error message should be present"
        assert "timeout" in str(result["error"]).lower(), \
            f"Error message should mention timeout, got: {result['error']}"

        # Verify timing: should be close to timeout, not the full sleep duration
        assert duration < 1.5, f"Execution took too long: {duration}s (expected < 1.5s)"
        assert duration >= 0.15, f"Execution too fast: {duration}s (expected >= 0.15s for 0.2s timeout)"
        assert result["runtime_seconds"] >= 0.15, \
            f"Runtime seconds too low: {result['runtime_seconds']}s"

        # Verify stdout/stderr contain what was printed before timeout
        assert "Starting sleep" in result["stderr"], "Should capture output before timeout"
        assert "Finished sleep" not in result["stderr"], "Should not capture output after timeout"

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

        # Verify memory limit was enforced
        assert result["exit_code"] != 0, \
            f"Memory limit should cause failure, but exit_code was {result['exit_code']}"
        
        # At least one of these should be true
        memory_detected = (
            result["memory_exceeded"]
            or "kill" in str(result["error"]).lower()
            or "MemoryError" in result["stderr"]
            or "memory" in str(result["error"]).lower()
        )
        assert memory_detected, \
            f"Memory limit violation not detected. error={result['error']}, " \
            f"memory_exceeded={result['memory_exceeded']}, stderr={result['stderr'][:200]}"

        # Verify error message is informative
        assert result["error"] is not None, "Error message should be present"
        
        # Verify stderr contains allocation attempt
        assert "Allocating memory" in result["stderr"], \
            "Should capture stderr before memory limit hit"

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

        # Verify successful execution
        assert result["exit_code"] == 0, f"Expected exit code 0, got {result['exit_code']}"
        assert result["error"] is None, f"Expected no error, got: {result['error']}"

        # Verify all output was captured
        assert len(result["stdout"]) > 10000, \
            f"Output too short: {len(result['stdout'])} chars (expected > 10000)"
        assert "Line 0" in result["stdout"], "Should contain first line"
        assert "Line 9999" in result["stdout"], "Should contain last line"
        assert "Line 5000" in result["stdout"], "Should contain middle line"

        # Verify output contains expected number of lines (approximately)
        line_count = result["stdout"].count("Line ")
        assert line_count == 10000, \
            f"Expected 10000 lines, found {line_count} occurrences of 'Line '"

        # Verify runtime is reasonable
        assert result["runtime_seconds"] > 0, "Runtime should be positive"
        assert result["runtime_seconds"] < 30.0, \
            f"Runtime too long for simple print loop: {result['runtime_seconds']}s"

    def test_syntax_error_reporting(self, tmp_path):
        """Test that syntax errors are correctly reported."""
        code = "this is not valid python"
        result = run_simulation(code=code, stage_id="syntax", output_dir=tmp_path)

        # Verify failure
        assert result["exit_code"] != 0, \
            f"Syntax error should cause non-zero exit, got {result['exit_code']}"
        assert result["error"] is not None, "Error message should be present"

        # Verify syntax error is detected and reported
        assert "SyntaxError" in result["stderr"], \
            f"stderr should contain 'SyntaxError', got: {result['stderr'][:200]}"
        
        # Verify error message mentions failure
        assert "failed" in str(result["error"]).lower() or \
               result["exit_code"] != 0, \
            f"Error message should indicate failure: {result['error']}"

        # Verify stdout is empty or minimal for syntax errors
        assert len(result["stdout"]) < 100, \
            f"stdout should be minimal for syntax errors, got: {result['stdout'][:100]}"

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

            # Verify signal kill is detected
            assert result["exit_code"] == -9, \
                f"Expected exit code -9 (SIGKILL), got {result['exit_code']}"
            assert result["error"] is not None, "Error message should be present"
            
            error_str = str(result["error"]).lower()
            assert "killed" in error_str or "signal" in error_str, \
                f"Error should mention kill/signal, got: {result['error']}"
            
            # Verify error message contains signal number
            assert "9" in error_str or "sigkill" in error_str, \
                f"Error should mention signal 9, got: {result['error']}"

    def test_empty_code_handling(self, tmp_path):
        """Test that empty code is handled correctly."""
        result = run_simulation(code="", stage_id="empty", output_dir=tmp_path)

        # Empty code should execute successfully (just does nothing)
        assert result["exit_code"] == 0, \
            f"Empty code should exit successfully, got {result['exit_code']}"
        assert result["error"] is None, \
            f"Empty code should not produce error, got: {result['error']}"
        assert result["stdout"] == "", "stdout should be empty"
        assert result["stderr"] == "", "stderr should be empty"

    def test_none_output_dir_creates_temp(self):
        """Test that None output_dir creates a temporary directory."""
        result = run_simulation(
            code="print('test')",
            stage_id="temp_dir_test",
            output_dir=None,
        )

        # Should succeed
        assert result["exit_code"] == 0, "Should execute successfully"
        assert result["error"] is None, "Should not have error"
        assert "test" in result["stdout"], "Should capture output"

    def test_invalid_config_values(self, tmp_path):
        """Test handling of invalid config values."""
        # Test negative timeout
        result = run_simulation(
            code="print('test')",
            stage_id="neg_timeout",
            output_dir=tmp_path,
            config={"timeout_seconds": -1},
        )
        # Should either fail gracefully or use default
        assert "exit_code" in result, "Should return result dict"
        assert "error" in result, "Should have error field"

        # Test zero memory limit
        result = run_simulation(
            code="print('test')",
            stage_id="zero_mem",
            output_dir=tmp_path,
            config={"max_memory_gb": 0.0},
        )
        assert "exit_code" in result, "Should return result dict"

        # Test negative memory
        result = run_simulation(
            code="print('test')",
            stage_id="neg_mem",
            output_dir=tmp_path,
            config={"max_memory_gb": -1.0},
        )
        assert "exit_code" in result, "Should return result dict"

    def test_output_file_listing(self, tmp_path):
        """Test that output files are correctly listed."""
        code = """
import numpy as np
np.save("test_output.npy", np.array([1, 2, 3]))
with open("test_output.txt", "w") as f:
    f.write("test content")
"""
        result = run_simulation(
            code=code,
            stage_id="file_listing",
            output_dir=tmp_path,
        )

        assert result["exit_code"] == 0, "Should execute successfully"
        assert isinstance(result["output_files"], list), \
            "output_files should be a list"
        
        # Verify files are listed (excluding script)
        file_names = [f for f in result["output_files"]]
        assert "test_output.npy" in file_names, \
            f"Should list test_output.npy, got: {result['output_files']}"
        assert "test_output.txt" in file_names, \
            f"Should list test_output.txt, got: {result['output_files']}"
        assert "simulation_file_listing.py" not in file_names, \
            "Should not list script file"

        # Verify files actually exist
        assert (tmp_path / "test_output.npy").exists(), "File should exist"
        assert (tmp_path / "test_output.txt").exists(), "File should exist"

    def test_script_file_creation(self, tmp_path):
        """Test that script file is created correctly."""
        code = "print('hello world')"
        stage_id = "script_test"
        
        result = run_simulation(
            code=code,
            stage_id=stage_id,
            output_dir=tmp_path,
        )

        script_path = tmp_path / f"simulation_{stage_id}.py"
        assert script_path.exists(), "Script file should exist"
        
        # Verify script content matches
        script_content = script_path.read_text()
        assert code in script_content, "Script content should match input code"

    def test_script_cleanup_when_keep_script_false(self, tmp_path):
        """Test that script is deleted when keep_script=False."""
        code = "print('test')"
        stage_id = "cleanup_test"
        
        result = run_simulation(
            code=code,
            stage_id=stage_id,
            output_dir=tmp_path,
            config={"keep_script": False},
        )

        script_path = tmp_path / f"simulation_{stage_id}.py"
        # Script should be deleted after execution
        assert not script_path.exists(), \
            f"Script should be deleted when keep_script=False, but exists at {script_path}"

    def test_script_persists_when_keep_script_true(self, tmp_path):
        """Test that script persists when keep_script=True (default)."""
        code = "print('test')"
        stage_id = "persist_test"
        
        result = run_simulation(
            code=code,
            stage_id=stage_id,
            output_dir=tmp_path,
            config={"keep_script": True},
        )

        script_path = tmp_path / f"simulation_{stage_id}.py"
        assert script_path.exists(), "Script should exist when keep_script=True"

    def test_runtime_seconds_tracking(self, tmp_path):
        """Test that runtime_seconds is accurately tracked."""
        code = """
import time
time.sleep(0.1)
print("done")
"""
        start = time.time()
        result = run_simulation(
            code=code,
            stage_id="runtime_test",
            output_dir=tmp_path,
        )
        wall_time = time.time() - start

        # Runtime should be positive and reasonable
        assert result["runtime_seconds"] > 0, \
            f"Runtime should be positive, got {result['runtime_seconds']}"
        assert result["runtime_seconds"] >= 0.08, \
            f"Runtime should be at least 0.08s, got {result['runtime_seconds']}"
        assert result["runtime_seconds"] <= wall_time + 1.0, \
            f"Runtime {result['runtime_seconds']}s should not exceed wall time {wall_time}s by much"

    def test_environment_variables_set(self, tmp_path):
        """Test that environment variables are set correctly."""
        code = """
import os
print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'NOT_SET')}")
print(f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', 'NOT_SET')}")
"""
        result = run_simulation(
            code=code,
            stage_id="env_test",
            output_dir=tmp_path,
            config={"max_cpu_cores": 2},
        )

        assert result["exit_code"] == 0, "Should execute successfully"
        assert "OMP_NUM_THREADS=2" in result["stdout"], \
            f"Should set OMP_NUM_THREADS=2, got: {result['stdout']}"
        assert "MKL_NUM_THREADS=2" in result["stdout"], \
            f"Should set MKL_NUM_THREADS=2, got: {result['stdout']}"

    def test_custom_environment_variables(self, tmp_path):
        """Test that custom environment variables are passed through."""
        code = """
import os
print(f"CUSTOM_VAR={os.environ.get('CUSTOM_VAR', 'NOT_SET')}")
"""
        result = run_simulation(
            code=code,
            stage_id="custom_env",
            output_dir=tmp_path,
            config={"env_vars": {"CUSTOM_VAR": "test_value"}},
        )

        assert result["exit_code"] == 0, "Should execute successfully"
        assert "CUSTOM_VAR=test_value" in result["stdout"], \
            f"Should set custom env var, got: {result['stdout']}"

    def test_import_error_handling(self, tmp_path):
        """Test that import errors are correctly reported."""
        code = "import nonexistent_module_xyz123"
        result = run_simulation(
            code=code,
            stage_id="import_error",
            output_dir=tmp_path,
        )

        assert result["exit_code"] != 0, "Import error should cause failure"
        assert result["error"] is not None, "Error message should be present"
        assert "ModuleNotFoundError" in result["stderr"] or \
               "ImportError" in result["stderr"], \
            f"Should report import error, got: {result['stderr'][:200]}"

    def test_runtime_error_handling(self, tmp_path):
        """Test that runtime errors are correctly reported."""
        code = "x = 1 / 0"
        result = run_simulation(
            code=code,
            stage_id="runtime_error",
            output_dir=tmp_path,
        )

        assert result["exit_code"] != 0, "Runtime error should cause failure"
        assert result["error"] is not None, "Error message should be present"
        assert "ZeroDivisionError" in result["stderr"], \
            f"Should report ZeroDivisionError, got: {result['stderr'][:200]}"

    def test_stderr_capture(self, tmp_path):
        """Test that stderr is correctly captured."""
        code = """
import sys
print("stdout message", file=sys.stdout)
print("stderr message", file=sys.stderr)
"""
        result = run_simulation(
            code=code,
            stage_id="stderr_test",
            output_dir=tmp_path,
        )

        assert result["exit_code"] == 0, "Should execute successfully"
        assert "stdout message" in result["stdout"], \
            f"Should capture stdout, got: {result['stdout']}"
        assert "stderr message" in result["stderr"], \
            f"Should capture stderr, got: {result['stderr']}"

    def test_very_short_timeout(self, tmp_path):
        """Test behavior with very short timeout."""
        code = "print('test')"
        result = run_simulation(
            code=code,
            stage_id="short_timeout",
            output_dir=tmp_path,
            config={"timeout_seconds": 0.1},
        )

        # Very short timeout might cause issues, but should not crash
        assert "exit_code" in result, "Should return result dict"
        assert "timeout_exceeded" in result, "Should have timeout_exceeded field"
        
        # If timeout was too short, it should be marked as exceeded
        # Otherwise, should succeed
        if result["timeout_exceeded"]:
            assert result["exit_code"] != 0, \
                "If timeout exceeded, exit code should be non-zero"

    def test_very_large_timeout(self, tmp_path):
        """Test behavior with very large timeout."""
        code = "print('test')"
        result = run_simulation(
            code=code,
            stage_id="large_timeout",
            output_dir=tmp_path,
            config={"timeout_seconds": 86400},  # 24 hours
        )

        assert result["exit_code"] == 0, "Should execute successfully"
        assert result["timeout_exceeded"] is False, "Should not timeout"

    def test_nan_in_output_does_not_cause_false_positive(self, tmp_path):
        """Test that printing np.nan does NOT cause false positive errors.
        
        We intentionally removed string-based NaN detection because it caused
        false positives from legitimate physics terms like "ε_inf" (epsilon infinity),
        "nanoantenna", variable names, etc.
        
        NaN/Inf detection is now handled by:
        1. Meep itself raises RuntimeError on field divergence (exit_code != 0)
        2. ExecutionValidatorAgent checks actual output files for NaN/Inf values
        3. Generated code should validate results and sys.exit(1) if NaN found
        """
        code = """
import numpy as np
print("Result:", np.nan)
print("eps_inf = 2.56")  # Legitimate physics term containing "inf"
"""
        result = run_simulation(
            code=code,
            stage_id="nan_test",
            output_dir=tmp_path,
        )

        # Should complete successfully - no false positive from "nan" string
        assert result["exit_code"] == 0, "Should not fail just because 'nan' appears in output"
        assert result["error"] is None, \
            f"Should not set error for legitimate 'nan' in output: {result['error']}"

    def test_inf_in_output_does_not_cause_false_positive(self, tmp_path):
        """Test that printing np.inf or physics terms like eps_inf does NOT cause errors.
        
        The string "inf" appears in many legitimate contexts:
        - np.inf, float('inf')
        - eps_inf (epsilon infinity - high frequency permittivity)
        - variable names containing "inf"
        """
        code = """
import numpy as np
print("Result:", np.inf)
print("eps_inf = 2.56")  # Common physics parameter name
print("Information: test complete")  # Word containing "inf"
"""
        result = run_simulation(
            code=code,
            stage_id="inf_test",
            output_dir=tmp_path,
        )

        # Should complete successfully - no false positive from "inf" string
        assert result["exit_code"] == 0, "Should not fail just because 'inf' appears in output"
        assert result["error"] is None, \
            f"Should not set error for legitimate 'inf' in output: {result['error']}"

    def test_output_dir_creation(self):
        """Test that output directory is created if it doesn't exist."""
        import tempfile
        temp_base = tempfile.gettempdir()
        new_dir = Path(temp_base) / "test_output_dir_creation_xyz"
        
        # Ensure it doesn't exist
        if new_dir.exists():
            import shutil
            shutil.rmtree(new_dir)

        result = run_simulation(
            code="print('test')",
            stage_id="dir_creation",
            output_dir=new_dir,
        )

        assert new_dir.exists(), "Output directory should be created"
        assert result["exit_code"] == 0, "Should execute successfully"

        # Cleanup
        import shutil
        if new_dir.exists():
            shutil.rmtree(new_dir)

    def test_materials_directory_symlink(self, tmp_path):
        """Test that materials directory is symlinked/copied if it exists."""
        # Create a fake materials directory in project root
        project_root = Path.cwd()
        materials_src = project_root / "materials"
        
        # Only test if materials directory exists
        if materials_src.exists():
            result = run_simulation(
                code="""
import os
if os.path.exists("materials"):
    print("materials directory exists")
    if os.path.islink("materials"):
        print("materials is a symlink")
    else:
        print("materials is a directory")
else:
    print("materials directory not found")
""",
                stage_id="materials_test",
                output_dir=tmp_path,
            )

            assert result["exit_code"] == 0, "Should execute successfully"
            assert "materials directory exists" in result["stdout"], \
                "Materials directory should be accessible"

    def test_validate_code_function(self):
        """Test the validate_code function."""
        # Test dangerous patterns
        dangerous_code = "os.system('rm -rf /')"
        warnings = validate_code(dangerous_code)
        assert len(warnings) > 0, "Should detect dangerous patterns"
        assert any("os.system" in w for w in warnings), \
            f"Should warn about os.system, got: {warnings}"

        # Test blocking patterns
        blocking_code = "input('Enter value:')"
        warnings = validate_code(blocking_code)
        assert len(warnings) > 0, "Should detect blocking patterns"
        assert any("BLOCKING" in w for w in warnings), \
            f"Should mark blocking patterns, got: {warnings}"

        # Test safe code
        safe_code = "import meep as mp\nprint('hello')"
        warnings = validate_code(safe_code)
        # Should have no warnings or only note about meep import
        dangerous_warnings = [w for w in warnings if "WARNING" in w or "BLOCKING" in w]
        assert len(dangerous_warnings) == 0, \
            f"Safe code should not have dangerous warnings, got: {warnings}"

    def test_estimate_runtime_function(self):
        """Test the estimate_runtime function."""
        # Test 2D code
        code_2d = "import meep as mp\nsim = mp.Simulation(cell_size=mp.Vector3(10, 10, 0))"
        estimate = estimate_runtime(code_2d)
        assert "estimated_minutes" in estimate, "Should return estimate"
        assert estimate["estimated_minutes"] > 0, "Estimate should be positive"
        assert estimate["features_detected"]["is_3d"] is False, "Should detect 2D"

        # Test 3D code
        code_3d = "import meep as mp\nv1 = mp.Vector3(1, 2, 3)\nv2 = mp.Vector3(4, 5, 6)\nv3 = mp.Vector3(7, 8, 9)"
        estimate = estimate_runtime(code_3d)
        assert estimate["features_detected"]["is_3d"] is True, "Should detect 3D"
        assert estimate["estimated_minutes"] > estimate_runtime(code_2d)["estimated_minutes"], \
            "3D should have higher estimate than 2D"

        # Test with sweep
        code_sweep = "for i in range(10):\n    print(i)"
        estimate = estimate_runtime(code_sweep)
        assert estimate["features_detected"]["has_sweep"] is True, "Should detect sweep"

    def test_run_code_node_missing_code(self):
        """Test run_code_node with missing code."""
        state = {"current_stage_id": "test", "paper_id": "test_paper"}
        result = run_code_node(state)

        assert "run_error" in result, "Should return run_error"
        assert result["run_error"] is not None, "Should have error for missing code"
        assert "code" in result["run_error"].lower() or "provided" in result["run_error"].lower(), \
            f"Error should mention code, got: {result['run_error']}"

    def test_run_code_node_success(self, tmp_path, monkeypatch):
        """Test run_code_node with valid code."""
        # Change to tmp_path so that "outputs" directory is created there
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            state = {
                "code": "print('test')",
                "current_stage_id": "test_stage",
                "paper_id": "test_paper",
                "plan": {
                    "stages": [{"stage_id": "test_stage", "runtime_budget_minutes": 60}]
                },
                "runtime_config": {"max_memory_gb": 8.0, "max_cpu_cores": 4},
            }

            result = run_code_node(state)

            assert "stage_outputs" in result, "Should return stage_outputs"
            assert "run_error" in result, "Should return run_error"
            
            # If execution succeeded, check outputs
            if result["run_error"] is None:
                assert "stdout" in result["stage_outputs"], "Should have stdout"
                assert "exit_code" in result["stage_outputs"], "Should have exit_code"
                assert result["stage_outputs"]["exit_code"] == 0, \
                    f"Should succeed, got exit_code {result['stage_outputs']['exit_code']}"
                assert "test" in result["stage_outputs"]["stdout"], \
                    "Should capture stdout output"
            else:
                # If there's an error, verify it's informative
                assert len(result["run_error"]) > 0, "Error message should not be empty"
        finally:
            os.chdir(original_cwd)

    def test_run_code_node_blocking_pattern(self):
        """Test run_code_node rejects blocking patterns."""
        state = {
            "code": "input('Enter value:')",
            "current_stage_id": "test_stage",
            "paper_id": "test_paper",
            "plan": {
                "stages": [{"stage_id": "test_stage", "runtime_budget_minutes": 60}]
            },
        }

        result = run_code_node(state)

        assert "run_error" in result, "Should return run_error"
        assert result["run_error"] is not None, "Should reject blocking code"
        assert "blocking" in result["run_error"].lower(), \
            f"Error should mention blocking, got: {result['run_error']}"

    def test_platform_capabilities_structure(self):
        """Test that platform capabilities have correct structure."""
        caps = get_platform_capabilities()

        required_fields = [
            "platform",
            "memory_limiting_available",
            "process_group_kill_available",
            "preexec_fn_available",
            "is_wsl",
            "warnings",
            "recommended_action",
        ]

        for field in required_fields:
            assert field in caps, f"Platform capabilities should have '{field}' field"

        assert isinstance(caps["platform"], str), "platform should be string"
        assert isinstance(caps["memory_limiting_available"], bool), \
            "memory_limiting_available should be bool"
        assert isinstance(caps["warnings"], list), "warnings should be list"

    def test_result_structure_completeness(self, tmp_path):
        """Test that ExecutionResult has all required fields."""
        result = run_simulation(
            code="print('test')",
            stage_id="structure_test",
            output_dir=tmp_path,
        )

        required_fields = [
            "stdout",
            "stderr",
            "exit_code",
            "output_files",
            "runtime_seconds",
            "error",
            "memory_exceeded",
            "timeout_exceeded",
        ]

        for field in required_fields:
            assert field in result, \
                f"ExecutionResult should have '{field}' field, got keys: {list(result.keys())}"

        # Verify types
        assert isinstance(result["stdout"], str), "stdout should be string"
        assert isinstance(result["stderr"], str), "stderr should be string"
        assert isinstance(result["exit_code"], int), "exit_code should be int"
        assert isinstance(result["output_files"], list), "output_files should be list"
        assert isinstance(result["runtime_seconds"], (int, float)), \
            "runtime_seconds should be numeric"
        assert isinstance(result["memory_exceeded"], bool), "memory_exceeded should be bool"
        assert isinstance(result["timeout_exceeded"], bool), "timeout_exceeded should be bool"

    def test_special_characters_in_stage_id(self, tmp_path):
        """Test that special characters in stage_id are handled safely."""
        code = "print('test')"
        # Test various special characters that might cause issues
        special_ids = ["stage-with-dash", "stage_with_underscore", "stage.with.dot"]
        
        for stage_id in special_ids:
            result = run_simulation(
                code=code,
                stage_id=stage_id,
                output_dir=tmp_path,
            )
            assert result["exit_code"] == 0, \
                f"Should handle stage_id '{stage_id}', got exit_code {result['exit_code']}"
            
            # Verify script file was created with correct name
            script_path = tmp_path / f"simulation_{stage_id}.py"
            assert script_path.exists(), \
                f"Script should exist for stage_id '{stage_id}'"

    def test_very_long_code_string(self, tmp_path):
        """Test handling of very long code strings."""
        # Generate a very long code string (100KB)
        long_code = "print('start')\n" + "# " + "x" * (100 * 1024) + "\nprint('end')"
        
        result = run_simulation(
            code=long_code,
            stage_id="long_code",
            output_dir=tmp_path,
        )

        assert result["exit_code"] == 0, \
            f"Should handle long code, got exit_code {result['exit_code']}"
        assert "start" in result["stdout"], "Should execute beginning of code"
        assert "end" in result["stdout"], "Should execute end of code"

    def test_unicode_in_code(self, tmp_path):
        """Test handling of Unicode characters in code."""
        code = """
# Test Unicode: 测试 🚀 émoji
print("Unicode test: 测试 🚀 émoji")
"""
        result = run_simulation(
            code=code,
            stage_id="unicode_test",
            output_dir=tmp_path,
        )

        assert result["exit_code"] == 0, "Should handle Unicode in code"
        assert "Unicode test" in result["stdout"], "Should execute Unicode code"
        # Verify Unicode is preserved
        assert "测试" in result["stdout"] or "🚀" in result["stdout"], \
            "Should preserve Unicode characters"

    def test_unicode_in_stage_id(self, tmp_path):
        """Test handling of Unicode characters in stage_id."""
        code = "print('test')"
        result = run_simulation(
            code=code,
            stage_id="测试_stage",
            output_dir=tmp_path,
        )

        assert result["exit_code"] == 0, "Should handle Unicode in stage_id"
        # Script file should be created (may have encoding issues, but shouldn't crash)
        script_files = list(tmp_path.glob("simulation_*.py"))
        assert len(script_files) > 0, "Should create script file even with Unicode stage_id"

    def test_script_write_failure_handling(self, tmp_path):
        """Test handling when script file write fails."""
        # Make output directory read-only to cause write failure
        import stat
        original_mode = tmp_path.stat().st_mode
        
        try:
            # On Unix, make directory read-only
            tmp_path.chmod(stat.S_IRUSR | stat.S_IXUSR)  # Read and execute only
            
            result = run_simulation(
                code="print('test')",
                stage_id="write_fail",
                output_dir=tmp_path,
            )

            # Should return error result, not crash
            assert "error" in result, "Should return error field"
            assert result["error"] is not None, \
                f"Should have error for write failure, got: {result.get('error')}"
            assert "write" in str(result["error"]).lower() or \
                   "failed" in str(result["error"]).lower(), \
                f"Error should mention write failure, got: {result['error']}"
        finally:
            # Restore permissions
            tmp_path.chmod(original_mode)

    def test_nonexistent_output_dir_parent(self):
        """Test behavior when parent of output_dir doesn't exist."""
        import tempfile
        temp_base = Path(tempfile.gettempdir())
        nonexistent_dir = temp_base / "nonexistent_parent_xyz" / "child_dir"
        
        # Ensure parent doesn't exist
        if nonexistent_dir.parent.exists():
            import shutil
            shutil.rmtree(nonexistent_dir.parent)

        result = run_simulation(
            code="print('test')",
            stage_id="nonexistent_test",
            output_dir=nonexistent_dir,
        )

        # Should create parent directories
        assert nonexistent_dir.exists(), "Should create parent directories"
        assert result["exit_code"] == 0, "Should execute successfully"

        # Cleanup
        import shutil
        if nonexistent_dir.parent.exists():
            shutil.rmtree(nonexistent_dir.parent)

    def test_config_merging_with_defaults(self, tmp_path):
        """Test that config values are properly merged with defaults."""
        code = "print('test')"
        
        # Test partial config
        result1 = run_simulation(
            code=code,
            stage_id="partial_config",
            output_dir=tmp_path,
            config={"timeout_seconds": 100},
        )
        assert result1["exit_code"] == 0, "Should use partial config with defaults"
        
        # Test empty config
        result2 = run_simulation(
            code=code,
            stage_id="empty_config",
            output_dir=tmp_path,
            config={},
        )
        assert result2["exit_code"] == 0, "Should use all defaults"
        
        # Test None config
        result3 = run_simulation(
            code=code,
            stage_id="none_config",
            output_dir=tmp_path,
            config=None,
        )
        assert result3["exit_code"] == 0, "Should use all defaults when config is None"

    def test_output_files_sorted(self, tmp_path):
        """Test that output_files list is sorted."""
        code = """
import numpy as np
for i in [3, 1, 2]:
    np.save(f"file_{i}.npy", np.array([i]))
"""
        result = run_simulation(
            code=code,
            stage_id="sort_test",
            output_dir=tmp_path,
        )

        assert result["exit_code"] == 0, "Should execute successfully"
        assert isinstance(result["output_files"], list), "output_files should be list"
        
        # Verify files are sorted
        assert result["output_files"] == sorted(result["output_files"]), \
            f"output_files should be sorted, got: {result['output_files']}"

    def test_multiple_output_files(self, tmp_path):
        """Test handling of multiple output files."""
        code = """
import numpy as np
for i in range(5):
    np.save(f"output_{i}.npy", np.array([i]))
    with open(f"output_{i}.txt", "w") as f:
        f.write(f"text {i}")
"""
        result = run_simulation(
            code=code,
            stage_id="multi_file",
            output_dir=tmp_path,
        )

        assert result["exit_code"] == 0, "Should execute successfully"
        assert len(result["output_files"]) == 10, \
            f"Should list 10 files, got {len(result['output_files'])}: {result['output_files']}"
        
        # Verify all files are listed
        for i in range(5):
            assert f"output_{i}.npy" in result["output_files"], \
                f"Should list output_{i}.npy"
            assert f"output_{i}.txt" in result["output_files"], \
                f"Should list output_{i}.txt"

    def test_empty_output_files_list(self, tmp_path):
        """Test that output_files is empty list when no files created."""
        code = "print('no files')"
        result = run_simulation(
            code=code,
            stage_id="no_files",
            output_dir=tmp_path,
        )

        assert result["exit_code"] == 0, "Should execute successfully"
        assert isinstance(result["output_files"], list), "output_files should be list"
        assert len(result["output_files"]) == 0, \
            f"Should have empty output_files, got: {result['output_files']}"

    def test_error_result_structure(self, tmp_path):
        """Test that error results have correct structure."""
        code = "import nonexistent_module_xyz789"
        result = run_simulation(
            code=code,
            stage_id="error_structure",
            output_dir=tmp_path,
        )

        # Verify all required fields are present even in error case
        required_fields = [
            "stdout", "stderr", "exit_code", "output_files",
            "runtime_seconds", "error", "memory_exceeded", "timeout_exceeded"
        ]
        for field in required_fields:
            assert field in result, \
                f"Error result should have '{field}' field"

        # Verify error-specific values
        assert result["exit_code"] != 0, "Error should have non-zero exit code"
        assert result["error"] is not None, "Error message should be present"
        assert isinstance(result["output_files"], list), "output_files should be list"

    def test_timeout_error_structure(self, tmp_path):
        """Test that timeout error results have correct structure."""
        code = "import time; time.sleep(2)"
        result = run_simulation(
            code=code,
            stage_id="timeout_structure",
            output_dir=tmp_path,
            config={"timeout_seconds": 0.2},
        )

        assert result["timeout_exceeded"] is True, "Should mark timeout as exceeded"
        assert result["error"] is not None, "Error message should be present"
        assert "timeout" in str(result["error"]).lower(), \
            f"Error should mention timeout, got: {result['error']}"
        assert result["runtime_seconds"] > 0, "Should record runtime"
        assert result["runtime_seconds"] < 1.0, \
            f"Runtime should be close to timeout, got {result['runtime_seconds']}s"

    def test_timeout_with_output_capture(self, tmp_path):
        """Test that timeout correctly captures output that was printed before timeout."""
        code = """
import sys
import time
print("Output before timeout", file=sys.stdout)
print("Error before timeout", file=sys.stderr)
sys.stdout.flush()
sys.stderr.flush()
time.sleep(2)
print("Output after timeout", file=sys.stdout)
"""
        result = run_simulation(
            code=code,
            stage_id="timeout_output",
            output_dir=tmp_path,
            config={"timeout_seconds": 0.2},
        )

        assert result["timeout_exceeded"] is True, "Should mark timeout as exceeded"
        # Verify stdout/stderr are strings (not bytes) and contain pre-timeout output
        assert isinstance(result["stdout"], str), "stdout should be string"
        assert isinstance(result["stderr"], str), "stderr should be string"
        assert "Output before timeout" in result["stdout"], \
            f"Should capture stdout before timeout, got: {result['stdout'][:200]}"
        assert "Error before timeout" in result["stderr"], \
            f"Should capture stderr before timeout, got: {result['stderr'][:200]}"
        # Should not have output after timeout
        assert "Output after timeout" not in result["stdout"], \
            "Should not capture output after timeout"

