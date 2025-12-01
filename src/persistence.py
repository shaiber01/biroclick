"""
Disk Artifact Persistence Module

This module provides controlled access to disk artifacts while enforcing
the critical rule: NEVER read from disk for execution decisions.

═══════════════════════════════════════════════════════════════════════════════
DUAL CHECKPOINTING: WHY THIS MATTERS
═══════════════════════════════════════════════════════════════════════════════

The system uses TWO checkpointing mechanisms:

1. LangGraph MemorySaver (SOURCE OF TRUTH for execution)
   - Managed by LangGraph runtime
   - Used for resuming graph execution
   - Contains complete ReproState

2. Disk JSON Files (ARTIFACTS for humans)
   - Written by this module
   - Used for debugging, review, auditing
   - May be out of sync with LangGraph state

CRITICAL RULE: Nodes must NEVER read from disk for execution decisions.
Always use `state` passed by LangGraph.

This module enforces that rule by:
- Providing save_artifact() for writing (safe)
- Providing read_artifact_for_debugging() with warnings (for debugging only)
- Providing load_*_from_disk() that ALWAYS THROW (trap accidental misuse)

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════
# SAFE OPERATIONS: Writing artifacts to disk
# ═══════════════════════════════════════════════════════════════════════

def save_artifact(
    data: Dict[str, Any],
    path: Path,
    artifact_type: str,
    paper_id: Optional[str] = None
) -> Path:
    """
    Save state artifact to disk for human review/debugging.
    
    These files are OUTPUT ONLY. They should never be read for execution
    decisions - always use state passed by LangGraph.
    
    Args:
        data: Dictionary data to save
        path: Output file path
        artifact_type: Type of artifact (plan, progress, assumptions, etc.)
        paper_id: Optional paper ID for logging
        
    Returns:
        Path to saved file
        
    Example:
        >>> save_artifact(state["plan"], output_dir / "_artifact_plan.json", "plan")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata header
    artifact_data = {
        "_artifact_metadata": {
            "type": artifact_type,
            "paper_id": paper_id,
            "saved_at": datetime.now().isoformat(),
            "warning": "ARTIFACT ONLY - Do not read for execution decisions. Use LangGraph state.",
        },
        **data
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(artifact_data, f, indent=2, default=str)
    
    return path


def save_plan_artifact(plan: Dict[str, Any], paper_id: str, output_dir: Path) -> Path:
    """Save plan artifact to disk."""
    path = output_dir / paper_id / "_artifact_plan.json"
    return save_artifact(plan, path, "plan", paper_id)


def save_progress_artifact(progress: Dict[str, Any], paper_id: str, output_dir: Path) -> Path:
    """Save progress artifact to disk."""
    path = output_dir / paper_id / "_artifact_progress.json"
    return save_artifact(progress, path, "progress", paper_id)


def save_assumptions_artifact(assumptions: Dict[str, Any], paper_id: str, output_dir: Path) -> Path:
    """Save assumptions artifact to disk."""
    path = output_dir / paper_id / "_artifact_assumptions.json"
    return save_artifact(assumptions, path, "assumptions", paper_id)


# ═══════════════════════════════════════════════════════════════════════
# DEBUGGING ONLY: Reading artifacts with explicit warnings
# ═══════════════════════════════════════════════════════════════════════

def read_artifact_for_debugging(
    path: Path,
    caller: str = "unknown"
) -> Dict[str, Any]:
    """
    Read a disk artifact FOR DEBUGGING ONLY.
    
    ⚠️ WARNING: This data may be stale. Never use for execution decisions.
    Always use state passed by LangGraph.
    
    This function exists for:
    - Manual inspection during development
    - Post-mortem debugging after a run
    - Generating reports from completed runs
    
    It should NEVER be called from within a LangGraph node for execution logic.
    
    Args:
        path: Path to artifact file
        caller: Who is calling this (for audit trail)
        
    Returns:
        Parsed JSON data (may be stale!)
        
    Example:
        >>> # For debugging only - not in production code paths!
        >>> plan = read_artifact_for_debugging(
        ...     Path("outputs/paper_123/_artifact_plan.json"),
        ...     caller="manual_debug_session"
        ... )
    """
    path = Path(path)
    
    warnings.warn(
        f"⚠️ Reading disk artifact '{path}' for debugging (caller: {caller}). "
        f"This data may be OUT OF SYNC with LangGraph state. "
        f"Do NOT use for execution decisions - use state['...'] instead.",
        UserWarning,
        stacklevel=2
    )
    
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Log the access
    _log_artifact_access(path, caller, "read_for_debugging")
    
    return data


def _log_artifact_access(path: Path, caller: str, operation: str) -> None:
    """Log artifact access for audit trail (silent, non-blocking)."""
    try:
        log_dir = path.parent / "_artifact_access_log.txt"
        with open(log_dir, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} | {operation} | {caller} | {path}\n")
    except Exception:
        pass  # Non-critical logging, don't break on failure


# ═══════════════════════════════════════════════════════════════════════
# FORBIDDEN OPERATIONS: Trap accidental misuse
# ═══════════════════════════════════════════════════════════════════════
#
# These functions exist ONLY to catch mistakes. If someone tries to load
# plan/progress/assumptions from disk for execution, they'll get a clear
# error instead of subtle state divergence bugs.
# ═══════════════════════════════════════════════════════════════════════

class DiskReadForbiddenError(RuntimeError):
    """Raised when code attempts to read execution state from disk."""
    pass


def load_plan_from_disk(paper_id: str, output_dir: Path = Path("outputs")) -> None:
    """
    FORBIDDEN: Plan must come from state, not disk.
    
    This function exists to catch accidental misuse. It always raises an error.
    
    If you're seeing this error, you're trying to read the plan from disk
    during execution. This is forbidden because disk artifacts may be stale.
    
    Instead, use:
        state["plan"]  # From LangGraph state
        
    Raises:
        DiskReadForbiddenError: Always
    """
    raise DiskReadForbiddenError(
        f"\n"
        f"═══════════════════════════════════════════════════════════════════════\n"
        f"FORBIDDEN: Cannot load plan for '{paper_id}' from disk for execution.\n"
        f"═══════════════════════════════════════════════════════════════════════\n"
        f"\n"
        f"You attempted to read:\n"
        f"  {output_dir / paper_id / '_artifact_plan.json'}\n"
        f"\n"
        f"This is FORBIDDEN because disk artifacts may be out of sync with\n"
        f"LangGraph state. Silent divergence leads to hard-to-debug issues.\n"
        f"\n"
        f"SOLUTION: Use state passed by LangGraph:\n"
        f"  plan = state['plan']  # ← Correct\n"
        f"\n"
        f"If you need to read artifacts for debugging, use:\n"
        f"  from src.persistence import read_artifact_for_debugging\n"
        f"  plan = read_artifact_for_debugging(path, caller='my_debug_script')\n"
        f"\n"
        f"See docs/workflow.md 'Dual Checkpointing Strategy' for details.\n"
        f"═══════════════════════════════════════════════════════════════════════"
    )


def load_progress_from_disk(paper_id: str, output_dir: Path = Path("outputs")) -> None:
    """
    FORBIDDEN: Progress must come from state, not disk.
    
    Always raises DiskReadForbiddenError. See load_plan_from_disk for details.
    """
    raise DiskReadForbiddenError(
        f"\n"
        f"═══════════════════════════════════════════════════════════════════════\n"
        f"FORBIDDEN: Cannot load progress for '{paper_id}' from disk.\n"
        f"═══════════════════════════════════════════════════════════════════════\n"
        f"\n"
        f"SOLUTION: Use state['progress'] from LangGraph state.\n"
        f"\n"
        f"See docs/workflow.md 'Dual Checkpointing Strategy' for details.\n"
        f"═══════════════════════════════════════════════════════════════════════"
    )


def load_assumptions_from_disk(paper_id: str, output_dir: Path = Path("outputs")) -> None:
    """
    FORBIDDEN: Assumptions must come from state, not disk.
    
    Always raises DiskReadForbiddenError. See load_plan_from_disk for details.
    """
    raise DiskReadForbiddenError(
        f"\n"
        f"═══════════════════════════════════════════════════════════════════════\n"
        f"FORBIDDEN: Cannot load assumptions for '{paper_id}' from disk.\n"
        f"═══════════════════════════════════════════════════════════════════════\n"
        f"\n"
        f"SOLUTION: Use state['assumptions'] from LangGraph state.\n"
        f"\n"
        f"See docs/workflow.md 'Dual Checkpointing Strategy' for details.\n"
        f"═══════════════════════════════════════════════════════════════════════"
    )


# ═══════════════════════════════════════════════════════════════════════
# CHECKPOINT MANAGEMENT (for LangGraph checkpoints, not artifacts)
# ═══════════════════════════════════════════════════════════════════════
#
# Note: These work with LangGraph checkpoints, which ARE safe to read
# for resuming execution. They are different from artifact files.
# ═══════════════════════════════════════════════════════════════════════

def get_artifact_paths(paper_id: str, output_dir: Path = Path("outputs")) -> Dict[str, Path]:
    """
    Get paths to all artifact files for a paper.
    
    These paths are for REFERENCE ONLY. Do not read these files during
    execution - use LangGraph state instead.
    
    Args:
        paper_id: Paper identifier
        output_dir: Base output directory
        
    Returns:
        Dict mapping artifact type to file path
    """
    base = output_dir / paper_id
    return {
        "plan": base / "_artifact_plan.json",
        "progress": base / "_artifact_progress.json",
        "assumptions": base / "_artifact_assumptions.json",
        "checkpoints_dir": base / "checkpoints",
    }


def list_artifacts(paper_id: str, output_dir: Path = Path("outputs")) -> Dict[str, bool]:
    """
    Check which artifacts exist for a paper.
    
    Args:
        paper_id: Paper identifier
        output_dir: Base output directory
        
    Returns:
        Dict mapping artifact type to existence boolean
    """
    paths = get_artifact_paths(paper_id, output_dir)
    return {
        artifact_type: path.exists() if isinstance(path, Path) else path.is_dir()
        for artifact_type, path in paths.items()
    }

