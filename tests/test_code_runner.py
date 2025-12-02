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
    
    def test_detect_platform_platform_is_string(self):
        """Test that platform field is a recognized string."""
        caps = detect_platform()
        
        valid_platforms = ["windows", "wsl", "macos", "linux"]
        assert caps["platform"] in valid_platforms
    
    def test_get_platform_capabilities_cached(self):
        """Test that get_platform_capabilities returns cached result."""
        caps1 = get_platform_capabilities()
        caps2 = get_platform_capabilities()
        
        # Should return the same dict (cached)
        assert caps1 == caps2
    
    @patch.dict(os.environ, {"REPROLAB_SKIP_RESOURCE_LIMITS": "1"})
    def test_platform_warnings_suppressed_with_env_var(self):
        """Test that warnings are suppressed when env var is set."""
        # This primarily tests that the check doesn't raise when env var is set
        from src.code_runner import check_platform_and_warn
        
        # Should not raise any warnings (env var suppresses them)
        caps = check_platform_and_warn()
        assert caps is not None


# ═══════════════════════════════════════════════════════════════════════
# Code Validation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCodeValidation:
    """Tests for validate_code function."""
    
    def test_validate_code_detects_dangerous_os_system(self):
        """Test that os.system calls are flagged."""
        code = """
import os
os.system("rm -rf /")
"""
        warnings = validate_code(code)
        
        assert any("os.system" in w for w in warnings)
    
    def test_validate_code_detects_dangerous_subprocess(self):
        """Test that subprocess calls are flagged."""
        code = """
import subprocess
subprocess.call(["ls"])
subprocess.run(["whoami"])
"""
        warnings = validate_code(code)
        
        assert any("subprocess" in w for w in warnings)

    def test_validate_code_detects_popen(self):
        """Test that subprocess.Popen is flagged."""
        code = "subprocess.Popen(['ls'])"
        warnings = validate_code(code)
        assert any("subprocess" in w for w in warnings)
    
    def test_validate_code_detects_eval(self):
        """Test that eval() is flagged."""
        code = """
user_input = "print('hello')"
eval(user_input)
"""
        warnings = validate_code(code)
        
        assert any("eval" in w for w in warnings)
    
    def test_validate_code_detects_exec(self):
        """Test that exec() is flagged."""
        code = """
exec("import os")
"""
        warnings = validate_code(code)
        
        assert any("exec" in w for w in warnings)
    
    def test_validate_code_detects_blocking_plt_show(self):
        """Test that plt.show() is flagged as blocking."""
        code = """
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.show()
"""
        warnings = validate_code(code)
        
        assert any("plt.show()" in w for w in warnings)
        assert any("BLOCKING" in w for w in warnings)
    
    def test_validate_code_detects_blocking_input(self):
        """Test that input() is flagged as blocking."""
        code = """
name = input("Enter name: ")
"""
        warnings = validate_code(code)
        
        assert any("input(" in w for w in warnings)
        assert any("BLOCKING" in w for w in warnings)
    
    def test_validate_code_detects_missing_meep_import(self):
        """Test that missing meep import is noted."""
        code = """
import numpy as np
# No meep import
"""
        warnings = validate_code(code)
        
        assert any("meep" in w.lower() for w in warnings)
    
    def test_validate_code_passes_valid_meep_code(self):
        """Test that valid Meep code generates fewer warnings."""
        code = """
import meep as mp
import numpy as np

sim = mp.Simulation(
    cell_size=mp.Vector3(10, 10, 0),
    resolution=20,
)
sim.run(until=100)
np.save("output.npy", sim.get_array(component=mp.Ez))
"""
        warnings = validate_code(code)
        
        # Should have no WARNING or BLOCKING warnings
        critical_warnings = [w for w in warnings if "WARNING" in w or "BLOCKING" in w]
        assert len(critical_warnings) == 0


# ═══════════════════════════════════════════════════════════════════════
# Runtime Estimation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRuntimeEstimation:
    """Tests for estimate_runtime function."""
    
    def test_estimate_runtime_detects_3d_simulation(self):
        """Test that 3D simulations are detected and increase estimate."""
        code_2d = """
import meep as mp
cell = mp.Vector3(10, 10, 0)  # 2D
sim = mp.Simulation(cell_size=cell, resolution=20)
"""
        code_3d = """
import meep as mp
cell = mp.Vector3(10, 10, 10)  # 3D
geometry = [mp.Block(center=mp.Vector3(0, 0, 0), size=mp.Vector3(1, 1, 1))]
sources = [mp.Source(mp.ContinuousSource(frequency=0.15), 
                     component=mp.Ez, center=mp.Vector3(0, 0, 0))]
sim = mp.Simulation(cell_size=cell, resolution=20, geometry=geometry, sources=sources)
"""
        
        estimate_2d = estimate_runtime(code_2d)
        estimate_3d = estimate_runtime(code_3d)
        
        assert estimate_3d["features_detected"]["is_3d"] is True
        assert estimate_3d["estimated_minutes"] > estimate_2d["estimated_minutes"]
    
    def test_estimate_runtime_detects_sweeps(self):
        """Test that parameter sweeps are detected."""
        code_sweep = """
import meep as mp
import numpy as np

for freq in np.linspace(0.1, 0.5, 20):
    # Run simulation for each frequency
    pass
"""
        estimate = estimate_runtime(code_sweep)
        
        assert estimate["features_detected"]["has_sweep"] is True
    
    def test_estimate_runtime_detects_flux(self):
        """Test that flux regions are detected."""
        code_flux = """
import meep as mp
sim.add_flux(0.15, 0.1, 50, mp.FluxRegion(center=mp.Vector3(0, 0)))
"""
        estimate = estimate_runtime(code_flux)
        
        assert estimate["features_detected"]["has_flux"] is True
    
    def test_estimate_runtime_provides_reasonable_timeout(self):
        """Test that recommended timeout is at least 2x estimate."""
        code = """
import meep as mp
sim = mp.Simulation(cell_size=mp.Vector3(10, 10, 0), resolution=20)
sim.run(until=100)
"""
        estimate = estimate_runtime(code)
        
        # Timeout should be at least 2x estimated minutes (converted to seconds)
        expected_min_timeout = estimate["estimated_minutes"] * 60 * 2
        assert estimate["recommended_timeout_seconds"] >= expected_min_timeout
    
    def test_estimate_runtime_uses_provided_estimate(self):
        """Test that design_estimate_minutes is used as base."""
        code = "import meep"
        
        estimate = estimate_runtime(code, design_estimate_minutes=30.0)
        
        # Base should be 30 minutes
        assert estimate["estimated_minutes"] >= 30.0


# ═══════════════════════════════════════════════════════════════════════
# Error Result Helper Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMakeErrorResult:
    """Tests for _make_error_result helper function."""
    
    def test_make_error_result_sets_defaults_correctly(self):
        """Test that default values are set correctly."""
        result = _make_error_result(error="Test error")
        
        assert result["error"] == "Test error"
        assert result["stdout"] == ""
        assert result["stderr"] == ""
        assert result["exit_code"] == -1
        assert result["output_files"] == []
        assert result["runtime_seconds"] == 0.0
        assert result["memory_exceeded"] is False
        assert result["timeout_exceeded"] is False
    
    def test_make_error_result_custom_values(self):
        """Test that custom values override defaults."""
        result = _make_error_result(
            error="Memory exceeded",
            stdout="Some output",
            stderr="Error details",
            exit_code=137,
            runtime_seconds=45.5,
            memory_exceeded=True,
        )
        
        assert result["error"] == "Memory exceeded"
        assert result["stdout"] == "Some output"
        assert result["stderr"] == "Error details"
        assert result["exit_code"] == 137
        assert result["runtime_seconds"] == 45.5
        assert result["memory_exceeded"] is True
        assert result["timeout_exceeded"] is False
    
    def test_make_error_result_timeout_flag(self):
        """Test timeout flag is set correctly."""
        result = _make_error_result(
            error="Timeout exceeded",
            timeout_exceeded=True,
        )
        
        assert result["timeout_exceeded"] is True
        assert result["memory_exceeded"] is False
    
    def test_make_error_result_returns_execution_result_type(self):
        """Test that result conforms to ExecutionResult structure."""
        result = _make_error_result(error="Test")
        
        # Should have all required keys
        required_keys = [
            "stdout", "stderr", "exit_code", "output_files",
            "runtime_seconds", "error", "memory_exceeded", "timeout_exceeded"
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════════════════
# Execution Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRunSimulation:
    """Tests for run_simulation function."""
    
    def test_run_simulation_executes_code(self, tmp_path):
        """Test that it actually runs python code."""
        # Use a simple script that prints something and writes a file
        code = """
import sys
print("Hello stdout")
print("Hello stderr", file=sys.stderr)
with open("output.txt", "w") as f:
    f.write("content")
"""
        result = run_simulation(
            code=code,
            stage_id="test_exec",
            output_dir=tmp_path,
            config={"timeout_seconds": 10}
        )
        
        assert result["exit_code"] == 0
        assert "Hello stdout" in result["stdout"]
        assert "Hello stderr" in result["stderr"]
        assert "output.txt" in result["output_files"]
        assert (tmp_path / "output.txt").exists()
        assert (tmp_path / "output.txt").read_text() == "content"
        assert result["error"] is None

    def test_run_simulation_timeout(self, tmp_path):
        """Test timeout handling with actual subprocess."""
        # A script that sleeps longer than timeout
        code = """
import time
time.sleep(2)
"""
        # Set timeout to 1 second
        result = run_simulation(
            code=code,
            stage_id="test_timeout",
            output_dir=tmp_path,
            config={"timeout_seconds": 1}
        )
        
        assert result["timeout_exceeded"] is True
        assert "exceeded timeout" in result["error"]
        
    def test_run_simulation_cleanup_script(self, tmp_path):
        """Test that script is deleted if keep_script is False."""
        code = "print('hello')"
        run_simulation(
            code=code,
            stage_id="cleanup_test",
            output_dir=tmp_path,
            config={"keep_script": False}
        )
        
        script_path = tmp_path / "simulation_cleanup_test.py"
        assert not script_path.exists()

    def test_run_simulation_keep_script(self, tmp_path):
        """Test that script is kept if keep_script is True."""
        code = "print('hello')"
        run_simulation(
            code=code,
            stage_id="keep_test",
            output_dir=tmp_path,
            config={"keep_script": True}
        )
        
        script_path = tmp_path / "simulation_keep_test.py"
        assert script_path.exists()
        assert "print('hello')" in script_path.read_text(encoding='utf-8')

    @patch("src.code_runner.subprocess.run")
    def test_run_simulation_memory_error_detection(self, mock_run, tmp_path):
        """Test detection of memory errors from stderr."""
        # Mock a failed run with MemoryError in stderr
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Traceback... MemoryError: cannot allocate..."
        mock_run.return_value = mock_result
        
        code = "print('memory')"
        result = run_simulation(
            code=code,
            stage_id="mem_test",
            output_dir=tmp_path
        )
        
        assert result["memory_exceeded"] is True
        assert "Memory limit exceeded" in result["error"]

    @patch("src.code_runner.subprocess.run")
    def test_run_simulation_divergence_detection(self, mock_run, tmp_path):
        """Test detection of divergence (NaN/Inf)."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "some output nan values detected"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        code = "print('nan')"
        result = run_simulation(
            code=code,
            stage_id="div_test",
            output_dir=tmp_path
        )
        
        assert result["error"] is not None
        assert "diverged" in result["error"].lower()

    def test_run_simulation_symlink_materials(self, tmp_path):
        """Test that materials directory is symlinked if present in cwd."""
        # Create dummy materials dir in cwd (which we mock by changing cwd or creating it)
        # Since we can't easily change cwd for the whole process safely without side effects,
        # we can mock os.getcwd or ensure we run in a controlled env.
        # Instead, let's use a temp dir as the "project root" and mock os.getcwd
        
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "materials").mkdir()
        (project_root / "materials" / "mat.csv").touch()
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        with patch("os.getcwd", return_value=str(project_root)):
            run_simulation(
                code="print('sim')",
                stage_id="symlink_test",
                output_dir=output_dir
            )
            
        # Check if materials link exists in output_dir
        assert (output_dir / "materials").exists()
        assert (output_dir / "materials" / "mat.csv").exists()

    def test_run_simulation_capture_large_output(self, tmp_path):
        """Test correct capturing of large stdout/stderr without hanging."""
        # Script generates ~100KB of output
        code = """
import sys
for i in range(1000):
    print(f"stdout line {i} " * 5)
    print(f"stderr line {i} " * 5, file=sys.stderr)
"""
        result = run_simulation(
            code=code,
            stage_id="large_out",
            output_dir=tmp_path,
            config={"timeout_seconds": 5}
        )
        
        assert result["exit_code"] == 0
        assert len(result["stdout"]) > 50000
        assert len(result["stderr"]) > 50000
        assert "stdout line 999" in result["stdout"]
        assert "stderr line 999" in result["stderr"]

    def test_run_simulation_unicode_output(self, tmp_path):
        """Test correct handling of unicode in output."""
        code = """
print("Hello 🌍")
print("Euro: €")
"""
        result = run_simulation(
            code=code,
            stage_id="unicode",
            output_dir=tmp_path
        )
        
        assert result["exit_code"] == 0
        assert "Hello 🌍" in result["stdout"]
        assert "Euro: €" in result["stdout"]

    def test_run_simulation_syntax_error(self, tmp_path):
        """Test handling of syntax errors in code."""
        code = """
def broken_function()
    print("Missing colon above")
"""
        result = run_simulation(
            code=code,
            stage_id="syntax_err",
            output_dir=tmp_path
        )
        
        assert result["exit_code"] != 0
        assert "SyntaxError" in result["stderr"]
        assert "Simulation failed" in result["error"]

    def test_run_simulation_runtime_error(self, tmp_path):
        """Test handling of runtime exceptions."""
        code = """
raise ValueError("Something went wrong")
"""
        result = run_simulation(
            code=code,
            stage_id="runtime_err",
            output_dir=tmp_path
        )
        
        assert result["exit_code"] != 0
        assert "ValueError: Something went wrong" in result["stderr"]
        assert "Simulation failed" in result["error"]

    def test_run_simulation_env_vars(self, tmp_path):
        """Test that env_vars from config are passed to subprocess."""
        code = """
import os
print(f"MY_VAR={os.environ.get('MY_VAR')}")
"""
        result = run_simulation(
            code=code,
            stage_id="env_test",
            output_dir=tmp_path,
            config={"env_vars": {"MY_VAR": "test_value"}}
        )
        
        assert result["exit_code"] == 0
        assert "MY_VAR=test_value" in result["stdout"]


# ═══════════════════════════════════════════════════════════════════════
# Node Integration Tests
# ═══════════════════════════════════════════════════════════════════════

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
        assert kwargs["config"]["env_vars"] is None # Default should be None

    def test_run_code_node_missing_code(self):
        """Test that missing code returns error."""
        state = {"current_stage_id": "test"} # no 'code' key
        result = run_code_node(state)
        assert "No simulation code provided" in result["run_error"]
        assert result["stage_outputs"] == {}

    def test_run_code_node_blocking_validation(self):
        """Test that blocking code validation prevents execution."""
        state = {
            "code": "input('block')",
            "current_stage_id": "stage1"
        }
        
        result = run_code_node(state)
        
        assert "Code contains blocking patterns" in result["run_error"]
        assert "validation_warnings" in result["stage_outputs"]

    def test_run_code_node_propagates_run_error(self):
        """Test that execution errors are propagated to state."""
        state = {
            "code": "raise Exception('fail')",
            "current_stage_id": "stage1"
        }
        
        # Should run actual simulation which fails
        result = run_code_node(state)
        
        assert result["run_error"] is not None
        assert "Execution failed" in result["run_error"] or "exit code" in result["run_error"]

    def test_run_code_node_propagates_flags(self):
        """Test that memory_exceeded and timeout_exceeded flags are propagated."""
        with patch("src.code_runner.run_simulation") as mock_run:
            mock_run.return_value = {
                "stdout": "", "stderr": "", "exit_code": -1,
                "output_files": [], "runtime_seconds": 10.0,
                "error": "Timeout",
                "memory_exceeded": False, 
                "timeout_exceeded": True
            }
            
            state = {
                "code": "import meep",
                "current_stage_id": "stage1"
            }
            
            result = run_code_node(state)
            
            assert result["stage_outputs"]["timeout_exceeded"] is True
            assert result["stage_outputs"]["memory_exceeded"] is False

    def test_run_simulation_detects_signal_kill(self, tmp_path):
        """Test detection of process killed by signal (e.g. OOM killer)."""
        # We can't easily force a signal kill in a cross-platform way reliably in a unit test
        # without external tools, so we'll use a mock to verify the logic handles negative return codes.
        
        with patch("src.code_runner.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = -9  # SIGKILL
            mock_result.stdout = ""
            mock_result.stderr = "" # No "Killed" message usually
            mock_run.return_value = mock_result
            
            result = run_simulation(
                code="print('running')",
                stage_id="kill_test",
                output_dir=tmp_path
            )
            
            # The current implementation checks stderr for "killed", so this MIGHT fail if logic is buggy
            # This test asserts what SHOULD happen
            assert "killed" in str(result["error"]).lower() or "signal" in str(result["error"]).lower()
