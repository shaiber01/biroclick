"""
Code Runner - Sandboxed Execution for LLM-Generated Meep Simulations

This module provides subprocess-based sandboxed execution of Meep simulation code
with timeout and memory limits.

V1 Implementation Features:
- Subprocess isolation
- Configurable timeout
- Memory limits (Unix via resource module)
- Thread/core limits via environment variables
- Output capture and file listing

Platform Detection:
- At module import time, platform capabilities are detected and warnings emitted
- Windows and native macOS users are warned about missing resource limiting
- Set REPROLAB_SKIP_RESOURCE_LIMITS=1 to suppress these warnings

See docs/guidelines.md Section 14 for design rationale.
"""

import subprocess
import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import TypedDict, Optional, List, Dict, Any
from typing_extensions import NotRequired


# ═══════════════════════════════════════════════════════════════════════
# Type Definitions
# ═══════════════════════════════════════════════════════════════════════

class ExecutionResult(TypedDict):
    """Result of running a simulation."""
    stdout: str
    stderr: str
    exit_code: int
    output_files: List[str]
    runtime_seconds: float
    error: Optional[str]
    memory_exceeded: bool
    timeout_exceeded: bool


class ExecutionConfig(TypedDict, total=False):
    """Configuration for simulation execution."""
    timeout_seconds: int  # Maximum runtime (default: 3600 = 1 hour)
    max_memory_gb: float  # Maximum memory in GB (default: 8.0)
    max_cpu_cores: int  # Maximum CPU cores/threads (default: 4)
    working_dir: NotRequired[str]  # Working directory (default: temp dir)
    keep_script: bool  # Keep the script file after execution (default: True)
    env_vars: NotRequired[Dict[str, str]]  # Additional environment variables


# Default configuration
DEFAULT_CONFIG: ExecutionConfig = {
    "timeout_seconds": 3600,  # 1 hour
    "max_memory_gb": 8.0,
    "max_cpu_cores": 4,
    "keep_script": True,
}


# ═══════════════════════════════════════════════════════════════════════
# Platform Detection and Compatibility
# ═══════════════════════════════════════════════════════════════════════

import sys
import warnings


class PlatformCapabilities(TypedDict):
    """Describes what sandboxing features are available on this platform."""
    platform: str
    memory_limiting_available: bool
    process_group_kill_available: bool
    preexec_fn_available: bool
    is_wsl: bool
    warnings: List[str]
    recommended_action: Optional[str]


def detect_platform() -> PlatformCapabilities:
    """
    Detect the current platform and its sandboxing capabilities.
    
    Returns:
        PlatformCapabilities dict describing what features are available
        
    Example:
        caps = detect_platform()
        if not caps["memory_limiting_available"]:
            print(f"Warning: {caps['recommended_action']}")
    """
    platform_warnings: List[str] = []
    recommended_action: Optional[str] = None
    
    # Check for WSL
    is_wsl = False
    try:
        if hasattr(os, 'uname'):
            uname = os.uname()
            is_wsl = 'microsoft' in uname.release.lower() or 'wsl' in uname.release.lower()
    except Exception:
        pass
    
    # Windows Native
    if sys.platform == 'win32':
        platform_warnings.append(
            "Running on Windows native. Memory limiting is NOT available. "
            "Simulations can consume unlimited memory."
        )
        platform_warnings.append(
            "Process group signaling is limited. Timeout enforcement may not "
            "kill all child processes."
        )
        recommended_action = (
            "For full functionality, consider using WSL2 or Docker. "
            "See docs/guidelines.md Section 14 for setup instructions."
        )
        
        return PlatformCapabilities(
            platform="windows",
            memory_limiting_available=False,
            process_group_kill_available=False,
            preexec_fn_available=False,
            is_wsl=False,
            warnings=platform_warnings,
            recommended_action=recommended_action
        )
    
    # WSL2 (behaves like Linux)
    if is_wsl:
        return PlatformCapabilities(
            platform="wsl",
            memory_limiting_available=True,
            process_group_kill_available=True,
            preexec_fn_available=True,
            is_wsl=True,
            warnings=[],
            recommended_action=None
        )
    
    # macOS
    if sys.platform == 'darwin':
        # Check for Apple Silicon
        try:
            import platform as plat
            machine = plat.machine()
            if machine == 'arm64':
                platform_warnings.append(
                    "Running on Apple Silicon. Ensure Meep is installed for ARM64 "
                    "or via Rosetta for best compatibility."
                )
        except Exception:
            pass
        
        return PlatformCapabilities(
            platform="macos",
            memory_limiting_available=True,
            process_group_kill_available=True,
            preexec_fn_available=True,
            is_wsl=False,
            warnings=platform_warnings,
            recommended_action=None
        )
    
    # Linux
    return PlatformCapabilities(
        platform="linux",
        memory_limiting_available=True,
        process_group_kill_available=True,
        preexec_fn_available=True,
        is_wsl=False,
        warnings=[],
        recommended_action=None
    )


def check_platform_and_warn() -> PlatformCapabilities:
    """
    Check platform capabilities and emit warnings if needed.
    
    Call this at module load or before first simulation to inform users
    of any platform limitations.
    
    Returns:
        PlatformCapabilities dict
    """
    caps = detect_platform()
    
    # Check environment variable to suppress warnings
    suppress_warnings = os.environ.get("REPROLAB_SKIP_RESOURCE_LIMITS", "0") == "1"
    
    if caps["warnings"] and not suppress_warnings:
        for warning in caps["warnings"]:
            warnings.warn(warning, RuntimeWarning)
        
        if caps["recommended_action"]:
            warnings.warn(f"RECOMMENDED: {caps['recommended_action']}", RuntimeWarning)
    
    return caps


# Detect platform at module load time and emit warnings
_PLATFORM_CAPS: Optional[PlatformCapabilities] = None

# Emit platform warnings at module import time
# This ensures users are informed of platform limitations when the module is first loaded
_PLATFORM_CAPS = check_platform_and_warn()


def get_platform_capabilities() -> PlatformCapabilities:
    """Get cached platform capabilities (detected once at module load)."""
    global _PLATFORM_CAPS
    if _PLATFORM_CAPS is None:
        _PLATFORM_CAPS = check_platform_and_warn()
    return _PLATFORM_CAPS


# ═══════════════════════════════════════════════════════════════════════
# Resource Limiting (Unix)
# ═══════════════════════════════════════════════════════════════════════

def _create_resource_limiter(max_memory_gb: float):
    """
    Create a preexec function that sets resource limits.
    
    Only works on Unix systems. On Windows, returns None.
    
    Uses platform detection to determine if memory limiting is available.
    
    Args:
        max_memory_gb: Maximum memory in gigabytes
        
    Returns:
        Callable to set limits, or None if not supported on this platform
    """
    caps = get_platform_capabilities()
    
    if not caps["memory_limiting_available"]:
        # Windows or other platform without resource module
        # Warning already emitted by check_platform_and_warn()
        return None
    
    def set_limits():
        try:
            import resource
            
            # Set address space limit (soft and hard)
            max_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            
            # Ensure we don't try to exceed system hard limit (which raises ValueError)
            # And ensure soft <= hard
            new_soft = min(max_bytes, hard)
            new_hard = min(max_bytes, hard)
            
            # Debug print to help diagnose failures (will be captured in tests)
            # print(f"DEBUG: Setting RLIMIT_AS. Request: {max_bytes}, Current: ({soft}, {hard}), New: ({new_soft}, {new_hard})")
            
            resource.setrlimit(resource.RLIMIT_AS, (new_soft, new_hard))
            
            # Optionally set other limits for more comprehensive control
            # resource.setrlimit(resource.RLIMIT_DATA, (max_bytes, max_bytes))
            # resource.setrlimit(resource.RLIMIT_RSS, (max_bytes, max_bytes))
            
        except (ImportError, ValueError) as e:
            # Some limits may not be available on all systems
            print(f"Warning: Could not set resource limits: {e}")
        except Exception as e:
            # resource.error or other platform-specific errors
            if "resource" in str(type(e).__module__):
                print(f"Warning: Resource limit error: {e}")
            else:
                raise
    
    return set_limits


# ═══════════════════════════════════════════════════════════════════════
# Environment Setup
# ═══════════════════════════════════════════════════════════════════════

def _create_execution_environment(
    max_cpu_cores: int,
    extra_env: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Create environment variables for controlled execution.
    
    Sets thread limits for common numerical libraries to prevent
    runaway parallelism.
    """
    env = os.environ.copy()
    
    # Limit threads for common numerical libraries
    thread_limit = str(max_cpu_cores)
    env.update({
        "OMP_NUM_THREADS": thread_limit,       # OpenMP
        "OPENBLAS_NUM_THREADS": thread_limit,  # OpenBLAS
        "MKL_NUM_THREADS": thread_limit,       # Intel MKL
        "NUMEXPR_NUM_THREADS": thread_limit,   # NumExpr
        "VECLIB_MAXIMUM_THREADS": thread_limit,  # macOS Accelerate
    })
    
    # Meep-specific settings
    env.update({
        "MEEP_PROGRESS": "1",  # Enable progress output
    })
    
    # Add any extra environment variables
    if extra_env:
        env.update(extra_env)
    
    return env


# ═══════════════════════════════════════════════════════════════════════
# Main Execution Functions
# ═══════════════════════════════════════════════════════════════════════

def run_simulation(
    code: str,
    stage_id: str,
    output_dir: Optional[Path] = None,
    config: Optional[ExecutionConfig] = None
) -> ExecutionResult:
    """
    Execute Meep simulation code in a sandboxed subprocess.
    
    Args:
        code: Python+Meep simulation code to execute
        stage_id: Unique identifier for this execution (used for script naming)
        output_dir: Directory for output files. If None, creates temp directory.
        config: Execution configuration (timeout, memory, etc.)
        
    Returns:
        ExecutionResult with stdout, stderr, exit_code, output_files, etc.
        
    Example:
        result = run_simulation(
            code=simulation_code,
            stage_id="stage_1_material",
            output_dir=Path("outputs/paper_123/stage_1"),
            config={"timeout_seconds": 1800, "max_memory_gb": 4.0}
        )
        
        if result["exit_code"] == 0:
            print("Success! Output files:", result["output_files"])
        else:
            print("Failed:", result["error"])
    """
    # Merge config with defaults
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    
    # Setup output directory
    temp_dir_created = False
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix=f"meep_{stage_id}_"))
        temp_dir_created = True
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write simulation code to file
    script_name = f"simulation_{stage_id}.py"
    script_path = output_dir / script_name
    
    try:
        script_path.write_text(code, encoding='utf-8')
        
        # Symlink materials directory if it exists in project root
        # This allows code to reference "materials/file.csv" naturally
        project_root = Path(os.getcwd())
        materials_src = project_root / "materials"
        materials_dst = output_dir / "materials"
        
        if materials_src.exists() and not materials_dst.exists():
            try:
                os.symlink(materials_src, materials_dst)
            except OSError as e:
                # Fallback for Windows or permissions issues: copy directory
                import shutil
                try:
                    shutil.copytree(materials_src, materials_dst)
                except Exception as copy_err:
                    print(f"Warning: Could not link/copy materials: {copy_err}")
                    
    except Exception as e:
        return _make_error_result(
            error=f"Script write failed: {e}",
            stderr=f"Failed to write script: {e}",
        )
    
    # Setup execution
    preexec_fn = _create_resource_limiter(cfg["max_memory_gb"])
    env = _create_execution_environment(
        cfg["max_cpu_cores"],
        cfg.get("env_vars")
    )
    
    # Record start time
    start_time = time.time()
    
    # Execute
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=str(output_dir),
            timeout=cfg["timeout_seconds"],
            capture_output=True,
            text=True,
            errors='replace',  # Handle binary output gracefully
            preexec_fn=preexec_fn,
            env=env
        )
        
        runtime_seconds = time.time() - start_time
        
        # Determine if there was an error
        error_msg = None
        if result.returncode != 0:
            # Check for common error patterns
            stderr_lower = result.stderr.lower()
            if "memoryerror" in stderr_lower or "cannot allocate" in stderr_lower:
                error_msg = "Memory limit exceeded"
            elif "killed" in stderr_lower or result.returncode < 0:
                # Negative return code means terminated by signal
                # -9 = SIGKILL, -15 = SIGTERM, etc.
                error_msg = f"Process killed (signal {-result.returncode}) - likely resource limit"
            else:
                error_msg = f"Simulation failed with exit code {result.returncode}"
        # NOTE: We do NOT check for "nan"/"inf" strings in stdout/stderr here.
        # Reason: This causes false positives from legitimate physics terms like
        # "ε_inf" (epsilon infinity), "nanoantenna", variable names, etc.
        # 
        # Instead, NaN/Inf detection is handled by:
        # 1. Meep itself raises RuntimeError on field divergence (exit_code != 0)
        # 2. ExecutionValidatorAgent checks actual output files for NaN/Inf values
        # 3. Generated code should validate results and sys.exit(1) if NaN found
        #
        # See: prompts/execution_validator_agent.md section D (Data Integrity)
        
        # List output files (excluding the script itself)
        output_files = _list_output_files(output_dir, exclude=[script_name])
        
        return ExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            output_files=output_files,
            runtime_seconds=runtime_seconds,
            error=error_msg,
            memory_exceeded="memoryerror" in result.stderr.lower(),
            timeout_exceeded=False
        )
        
    except subprocess.TimeoutExpired as e:
        runtime_seconds = time.time() - start_time
        
        # Handle stdout/stderr - they may be bytes or strings depending on text mode
        # With text=True, they are strings; with text=False, they are bytes
        stdout_str = ""
        if e.stdout:
            if isinstance(e.stdout, bytes):
                stdout_str = e.stdout.decode(errors='replace')
            else:
                stdout_str = e.stdout
        
        stderr_str = ""
        if e.stderr:
            if isinstance(e.stderr, bytes):
                stderr_str = e.stderr.decode(errors='replace')
            else:
                stderr_str = e.stderr
        
        return _make_error_result(
            error=f"Simulation exceeded timeout ({cfg['timeout_seconds']}s)",
            stdout=stdout_str,
            stderr=stderr_str,
            output_dir=output_dir,
            exclude_files=[script_name],
            runtime_seconds=runtime_seconds,
            timeout_exceeded=True,
        )
        
    except MemoryError:
        runtime_seconds = time.time() - start_time
        
        return _make_error_result(
            error=f"Memory limit exceeded ({cfg['max_memory_gb']}GB)",
            stderr="MemoryError in subprocess management",
            output_dir=output_dir,
            exclude_files=[script_name],
            runtime_seconds=runtime_seconds,
            memory_exceeded=True,
        )
        
    except Exception as e:
        runtime_seconds = time.time() - start_time
        
        return _make_error_result(
            error=f"Execution failed: {e}",
            stderr=str(e),
            output_dir=output_dir,
            exclude_files=[script_name],
            runtime_seconds=runtime_seconds,
        )
    
    finally:
        # Cleanup script if not keeping it
        if not cfg.get("keep_script", True) and script_path.exists():
            try:
                script_path.unlink()
            except Exception:
                pass
        
        # Cleanup temp directory if we created it and execution failed
        if temp_dir_created and not _list_output_files(output_dir, exclude=[script_name]):
            try:
                shutil.rmtree(output_dir)
            except Exception:
                pass


def _list_output_files(
    directory: Path,
    exclude: Optional[List[str]] = None
) -> List[str]:
    """List all files in directory, excluding specified names."""
    exclude = exclude or []
    
    # Convert string to Path if needed (for robustness)
    if isinstance(directory, str):
        directory = Path(directory)
    
    try:
        files = []
        for f in directory.iterdir():
            if f.is_file() and f.name not in exclude:
                files.append(f.name)
        return sorted(files)
    except Exception:
        return []


def _make_error_result(
    error: str,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = -1,
    output_dir: Optional[Path] = None,
    exclude_files: Optional[List[str]] = None,
    runtime_seconds: float = 0.0,
    memory_exceeded: bool = False,
    timeout_exceeded: bool = False,
) -> ExecutionResult:
    """
    Construct an ExecutionResult for error cases.
    
    This helper reduces boilerplate in exception handlers by providing
    sensible defaults for error scenarios.
    
    Args:
        error: Error message describing what went wrong
        stdout: Captured stdout (default: empty)
        stderr: Captured stderr (default: empty)
        exit_code: Process exit code (default: -1)
        output_dir: Directory to scan for output files (default: None)
        exclude_files: Files to exclude from output listing
        runtime_seconds: How long execution ran before error
        memory_exceeded: Whether this was a memory limit error
        timeout_exceeded: Whether this was a timeout error
        
    Returns:
        ExecutionResult with appropriate fields set
    """
    output_files = []
    if output_dir is not None:
        output_files = _list_output_files(output_dir, exclude=exclude_files)
    
    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        output_files=output_files,
        runtime_seconds=runtime_seconds,
        error=error,
        memory_exceeded=memory_exceeded,
        timeout_exceeded=timeout_exceeded,
    )


# ═══════════════════════════════════════════════════════════════════════
# Validation and Utilities
# ═══════════════════════════════════════════════════════════════════════

def validate_code(code: str) -> List[str]:
    """
    Basic validation of simulation code before execution.
    
    Returns list of warnings/issues found.
    
    Args:
        code: String containing Python code to validate
        
    Raises:
        TypeError: If code is not a string
    """
    # Validate input type
    if not isinstance(code, str):
        raise TypeError(f"code must be a string, got {type(code).__name__}")
    
    warnings = []
    
    # Check for dangerous patterns
    dangerous_patterns = [
        ("os.system", "Potential shell command execution"),
        ("subprocess.call", "Potential subprocess execution"),
        ("subprocess.run", "Potential subprocess execution"),
        ("subprocess.Popen", "Potential subprocess execution"),
        ("eval(", "Potential code injection via eval"),
        ("exec(", "Potential code injection via exec"),
        ("__import__", "Dynamic import detected"),
        ("open('/etc", "Attempting to read system files"),
        ("open('/usr", "Attempting to read system files"),
        ("shutil.rmtree", "Attempting recursive file deletion"),
    ]
    
    for pattern, message in dangerous_patterns:
        if pattern in code:
            warnings.append(f"WARNING: {message} - found '{pattern}'")
    
    # Check for blocking patterns
    blocking_patterns = [
        ("plt.show()", "plt.show() will block headless execution"),
        ("input(", "input() will block automation"),
        ("raw_input(", "raw_input() will block automation"),
    ]
    
    for pattern, message in blocking_patterns:
        if pattern in code:
            warnings.append(f"BLOCKING: {message}")
    
    # Check for required imports - only check actual import statements, not comments or strings
    has_meep_import = False
    for line in code.split('\n'):
        # Strip whitespace and check if line is a comment
        stripped = line.strip()
        if stripped.startswith('#'):
            continue  # Skip comment lines
        
        # Check if line contains an actual import statement (not in a string)
        # Simple heuristic: check if "import meep" or "from meep" appears at start of line (after stripping)
        # This is not perfect but better than checking the whole code string
        if stripped.startswith('import meep') or stripped.startswith('from meep'):
            has_meep_import = True
            break
    
    if not has_meep_import:
        warnings.append("NOTE: No meep import found")
    
    return warnings


def estimate_runtime(
    code: str,
    design_estimate_minutes: Optional[float] = None
) -> Dict[str, Any]:
    """
    Estimate simulation runtime based on code analysis.
    
    This is a rough heuristic - actual runtime depends on hardware and problem size.
    """
    # Basic heuristics
    has_3d = "mp.Vector3" in code and code.count("mp.Vector3") > 2
    has_sweep = "for" in code and ("range(" in code or "np.linspace" in code)
    # Check for FluxRegion usage (mp.FluxRegion or FluxRegion() to avoid false positives in variable names)
    has_flux = ("mp.FluxRegion" in code or "FluxRegion(" in code or "add_flux" in code)
    # Check for Near2FarRegion usage (mp.Near2FarRegion or Near2FarRegion() to avoid false positives)
    has_near2far = ("mp.Near2FarRegion" in code or "Near2FarRegion(" in code or "add_near2far" in code)
    
    # Base estimate
    if design_estimate_minutes is not None:
        estimate = design_estimate_minutes
    else:
        estimate = 5.0  # Default 5 minutes
    
    # Adjust based on features
    if has_3d:
        estimate *= 10  # 3D is much slower
    if has_sweep:
        estimate *= 5  # Sweeps multiply runtime
    if has_near2far:
        estimate *= 2  # Near-to-far is computationally intensive
    
    return {
        "estimated_minutes": estimate,
        "recommended_timeout_seconds": int(estimate * 60 * 2),  # 2x buffer
        "features_detected": {
            "is_3d": has_3d,
            "has_sweep": has_sweep,
            "has_flux": has_flux,
            "has_near2far": has_near2far,
        }
    }


# ═══════════════════════════════════════════════════════════════════════
# LangGraph Node Integration
# ═══════════════════════════════════════════════════════════════════════

def run_code_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for RUN_CODE.
    
    Extracts configuration from state and executes the simulation.
    
    ARCHITECTURAL NOTE: This node returns raw execution results.
    It does NOT interpret failures or decide recovery strategies.
    
    FAILURE SEMANTICS ARE CENTRALIZED IN ExecutionValidatorAgent:
    - ExecutionValidatorAgent receives stage_outputs and run_error
    - It alone decides: regenerate code vs escalate vs fail stage
    - This keeps failure handling in one place, not scattered
    
    See prompts/execution_validator_agent.md for failure interpretation rules.
    
    Expected state fields:
        - code: str - The simulation code to execute
        - current_stage_id: str - Stage identifier
        - paper_id: str - Paper identifier
        - plan: dict - Contains stages with runtime_budget_minutes
        - runtime_config: dict (optional) - Hardware configuration
        
    Returns state updates:
        - stage_outputs: dict with stdout, stderr, files, etc.
        - run_error: str or None (raw error, not interpreted)
    """
    from pathlib import Path
    
    # Extract required fields
    code = state.get("code")
    if not code:
        return {
            "run_error": "No simulation code provided",
            "stage_outputs": {}
        }
    
    stage_id = state.get("current_stage_id", "unknown")
    paper_id = state.get("paper_id", "unknown")
    
    # Get runtime configuration
    plan = state.get("plan", {})
    stages = plan.get("stages", [])
    current_stage = next(
        (s for s in stages if s.get("stage_id") == stage_id),
        {}
    )
    
    runtime_budget_raw = current_stage.get("runtime_budget_minutes")
    # Use default if None or not a valid number type
    if runtime_budget_raw is None or not isinstance(runtime_budget_raw, (int, float)):
        runtime_budget = 60
    else:
        runtime_budget = runtime_budget_raw
    runtime_config = state.get("runtime_config", {})
    
    # Build output directory using run_output_dir if available, otherwise legacy path
    run_output_dir = state.get("run_output_dir", "")
    if run_output_dir:
        output_base = Path(run_output_dir) / stage_id
    else:
        # Legacy path for backwards compatibility
        output_base = Path("outputs") / paper_id / stage_id
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Build execution config
    config: ExecutionConfig = {
        "timeout_seconds": int(runtime_budget * 60),
        "max_memory_gb": runtime_config.get("max_memory_gb", 8.0),
        "max_cpu_cores": runtime_config.get("max_cpu_cores", 4),
        "keep_script": True,
    }
    
    # Validate code first
    warnings = validate_code(code)
    blocking_warnings = [w for w in warnings if w.startswith("BLOCKING")]
    if blocking_warnings:
        return {
            "run_error": f"Code contains blocking patterns: {blocking_warnings}",
            "stage_outputs": {"validation_warnings": warnings}
        }
    
    # Execute simulation
    result = run_simulation(
        code=code,
        stage_id=stage_id,
        output_dir=output_base,
        config=config
    )
    
    # Log execution result
    # Note: Use exit_code as primary success indicator, not error message.
    # Generated code may print warnings/errors but still produce valid output.
    import logging
    logger = logging.getLogger(__name__)
    runtime_str = f"{result['runtime_seconds']:.1f}s"
    num_files = len(result['output_files'])
    exit_code = result['exit_code']
    
    if exit_code != 0:
        # Non-zero exit code is definitive failure
        logger.info(f"⚡ run_code: stage={stage_id}, FAILED in {runtime_str} (exit={exit_code}: {result['error'] or 'unknown error'})")
    elif result['error']:
        # Exit code 0 but error message present - likely spurious, log as warning
        logger.info(f"⚡ run_code: stage={stage_id}, COMPLETED in {runtime_str}, {num_files} file(s) (warning: {result['error']})")
    else:
        logger.info(f"⚡ run_code: stage={stage_id}, SUCCESS in {runtime_str}, {num_files} output file(s)")
    
    # Return state updates
    return {
        "stage_outputs": {
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "exit_code": result["exit_code"],
            "files": result["output_files"],
            "runtime_seconds": result["runtime_seconds"],
            "validation_warnings": warnings,
            "timeout_exceeded": result["timeout_exceeded"],
            "memory_exceeded": result["memory_exceeded"],
        },
        "run_error": result["error"],
    }


# ═══════════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example simulation code
    example_code = '''
import meep as mp
import numpy as np

# Simple 2D simulation
resolution = 20
cell = mp.Vector3(10, 10, 0)

geometry = [
    mp.Block(
        center=mp.Vector3(0, 0),
        size=mp.Vector3(1, 1, mp.inf),
        material=mp.Medium(epsilon=12)
    )
]

sources = [
    mp.Source(
        mp.GaussianSource(frequency=0.15, fwidth=0.1),
        component=mp.Ez,
        center=mp.Vector3(-4, 0)
    )
]

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
    boundary_layers=[mp.PML(1.0)]
)

sim.run(until=200)

# Save field data
ez = sim.get_array(component=mp.Ez)
np.save("ez_field.npy", ez)
print(f"Saved field data with shape {ez.shape}")
'''
    
    print("=== Code Runner Test ===")
    print()
    
    # Validate code
    print("Validating code...")
    warnings = validate_code(example_code)
    if warnings:
        print("Warnings:", warnings)
    else:
        print("No warnings")
    
    # Estimate runtime
    print()
    print("Estimating runtime...")
    estimate = estimate_runtime(example_code)
    print(f"Estimated: {estimate['estimated_minutes']:.1f} minutes")
    print(f"Features: {estimate['features_detected']}")
    
    # Run simulation (uncomment to actually run)
    # print()
    # print("Running simulation...")
    # result = run_simulation(
    #     code=example_code,
    #     stage_id="test",
    #     config={"timeout_seconds": 300, "max_memory_gb": 2.0}
    # )
    # print(f"Exit code: {result['exit_code']}")
    # print(f"Runtime: {result['runtime_seconds']:.1f}s")
    # print(f"Output files: {result['output_files']}")
    # if result['error']:
    #     print(f"Error: {result['error']}")

