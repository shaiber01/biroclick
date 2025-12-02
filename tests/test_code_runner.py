"""
Tests for the Code Runner Module.

Tests platform detection, code validation, runtime estimation, and the
error result helper function.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.code_runner import (
    detect_platform,
    get_platform_capabilities,
    validate_code,
    estimate_runtime,
    _make_error_result,
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



