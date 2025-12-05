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


# ═══════════════════════════════════════════════════════════════════════
# JSON CHECKPOINT SAVER (for LangGraph persistence)
# ═══════════════════════════════════════════════════════════════════════
#
# This checkpointer persists LangGraph's execution state to JSON files,
# enabling resume from interrupts after process exit.
#
# Unlike MemorySaver (in-memory only), this survives process restarts.
# ═══════════════════════════════════════════════════════════════════════

import logging
import base64
from typing import Iterator, Tuple, List, Sequence
from contextlib import contextmanager

try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
        ChannelVersions,
        PendingWrite,
    )
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Define stub types for when langgraph is not available
    BaseCheckpointSaver = object
    Checkpoint = dict
    CheckpointMetadata = dict
    CheckpointTuple = tuple
    ChannelVersions = dict
    PendingWrite = tuple


class JsonCheckpointSaver(BaseCheckpointSaver):
    """
    JSON file-based checkpoint saver for LangGraph.
    
    Persists checkpoints to JSON files in a specified directory,
    enabling true resume from interrupts after process exit.
    
    Directory structure:
        {checkpoint_dir}/langgraph/
            checkpoint_{thread_id}_{checkpoint_id}.json
            writes_{thread_id}_{checkpoint_id}_{task_id}.json
    
    Usage:
        checkpointer = JsonCheckpointSaver("/path/to/run/checkpoints")
        graph = workflow.compile(checkpointer=checkpointer)
        
        # Resume later with same thread_id:
        graph.invoke(None, {"configurable": {"thread_id": "my_thread"}})
    """
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize JSON checkpoint saver.
        
        Args:
            checkpoint_dir: Directory to store checkpoints (e.g., run_output_dir/checkpoints)
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is required for JsonCheckpointSaver. "
                "Install with: pip install langgraph"
            )
        
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir) / "langgraph"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.serde = JsonPlusSerializer()
        self.logger = logging.getLogger(__name__)
    
    def _checkpoint_path(self, thread_id: str, checkpoint_id: str) -> Path:
        """Get path for a checkpoint file."""
        # Sanitize IDs for filesystem
        safe_thread = thread_id.replace("/", "_").replace("\\", "_")
        safe_checkpoint = checkpoint_id.replace("/", "_").replace("\\", "_")
        return self.checkpoint_dir / f"checkpoint_{safe_thread}_{safe_checkpoint}.json"
    
    def _writes_path(self, thread_id: str, checkpoint_id: str, task_id: str) -> Path:
        """Get path for a writes file."""
        safe_thread = thread_id.replace("/", "_").replace("\\", "_")
        safe_checkpoint = checkpoint_id.replace("/", "_").replace("\\", "_")
        safe_task = task_id.replace("/", "_").replace("\\", "_")
        return self.checkpoint_dir / f"writes_{safe_thread}_{safe_checkpoint}_{safe_task}.json"
    
    def _get_thread_checkpoints(self, thread_id: str) -> List[Path]:
        """Get all checkpoint files for a thread, sorted by modification time (newest first)."""
        safe_thread = thread_id.replace("/", "_").replace("\\", "_")
        pattern = f"checkpoint_{safe_thread}_*.json"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        return sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)
    
    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict:
        """
        Save a checkpoint to disk.
        
        Args:
            config: Configuration with thread_id and checkpoint_id
            checkpoint: The checkpoint data to save
            metadata: Checkpoint metadata (step, source, etc.)
            new_versions: Channel version updates
            
        Returns:
            Updated config with checkpoint_id
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
        
        # Serialize checkpoint data using msgpack, then base64 encode for JSON storage
        cp_type, cp_bytes = self.serde.dumps_typed(checkpoint)
        md_type, md_bytes = self.serde.dumps_typed(metadata)
        
        checkpoint_data = {
            "checkpoint_type": cp_type,
            "checkpoint_data": base64.b64encode(cp_bytes).decode("ascii"),
            "metadata_type": md_type,
            "metadata_data": base64.b64encode(md_bytes).decode("ascii"),
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "saved_at": datetime.now().isoformat(),
        }
        
        # Write to file
        filepath = self._checkpoint_path(thread_id, checkpoint_id)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.debug(f"Saved checkpoint: {filepath.name}")
        
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }
    
    def put_writes(
        self,
        config: dict,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        Save pending writes to disk.
        
        Args:
            config: Configuration with thread_id and checkpoint_id
            writes: List of (channel, value) tuples to write
            task_id: ID of the task that produced these writes
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id", "")
        
        if not checkpoint_id:
            return  # No checkpoint to associate writes with
        
        # Serialize writes using msgpack, then base64 encode
        serialized_writes = []
        for channel, value in writes:
            val_type, val_bytes = self.serde.dumps_typed(value)
            serialized_writes.append({
                "channel": channel,
                "value_type": val_type,
                "value_data": base64.b64encode(val_bytes).decode("ascii"),
            })
        
        writes_data = {
            "writes": serialized_writes,
            "task_id": task_id,
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "saved_at": datetime.now().isoformat(),
        }
        
        # Write to file
        filepath = self._writes_path(thread_id, checkpoint_id, task_id)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(writes_data, f, indent=2)
    
    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        """
        Retrieve a checkpoint from disk.
        
        Args:
            config: Configuration with thread_id and optionally checkpoint_id
            
        Returns:
            CheckpointTuple or None if not found
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")
        
        # If no specific checkpoint requested, get the latest
        if not checkpoint_id:
            checkpoints = self._get_thread_checkpoints(thread_id)
            if not checkpoints:
                return None
            filepath = checkpoints[0]  # Most recent
        else:
            filepath = self._checkpoint_path(thread_id, checkpoint_id)
            if not filepath.exists():
                return None
        
        # Load checkpoint data
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Deserialize using base64 decode then msgpack
        checkpoint = self.serde.loads_typed(
            (data["checkpoint_type"], base64.b64decode(data["checkpoint_data"]))
        )
        metadata = self.serde.loads_typed(
            (data["metadata_type"], base64.b64decode(data["metadata_data"]))
        )
        
        # Load pending writes if any
        pending_writes: List[PendingWrite] = []
        safe_thread = thread_id.replace('/', '_').replace('\\', '_')
        writes_pattern = f"writes_{safe_thread}_{data['checkpoint_id']}_*.json"
        for writes_file in self.checkpoint_dir.glob(writes_pattern):
            with open(writes_file, 'r', encoding='utf-8') as f:
                writes_data = json.load(f)
            for write in writes_data["writes"]:
                value = self.serde.loads_typed(
                    (write["value_type"], base64.b64decode(write["value_data"]))
                )
                pending_writes.append((writes_data["task_id"], write["channel"], value))
        
        # Build parent config if exists
        parent_config = None
        if data.get("parent_checkpoint_id"):
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": data["parent_checkpoint_id"],
                }
            }
        
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": data["checkpoint_id"],
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes if pending_writes else None,
        )
    
    def list(
        self,
        config: Optional[dict],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """
        List checkpoints matching the given criteria.
        
        Args:
            config: Configuration with thread_id
            filter: Optional filter criteria
            before: Optional config to list checkpoints before
            limit: Maximum number of checkpoints to return
            
        Yields:
            CheckpointTuple for each matching checkpoint
        """
        if config is None:
            return
        
        thread_id = config["configurable"]["thread_id"]
        checkpoints = self._get_thread_checkpoints(thread_id)
        
        count = 0
        for filepath in checkpoints:
            if limit is not None and count >= limit:
                break
            
            # Load and yield checkpoint
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # If 'before' specified, skip checkpoints at or after it
            if before:
                before_id = before["configurable"].get("checkpoint_id")
                if before_id and data["checkpoint_id"] >= before_id:
                    continue
            
            checkpoint = self.serde.loads_typed(
                (data["checkpoint_type"], base64.b64decode(data["checkpoint_data"]))
            )
            metadata = self.serde.loads_typed(
                (data["metadata_type"], base64.b64decode(data["metadata_data"]))
            )
            
            parent_config = None
            if data.get("parent_checkpoint_id"):
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": data["parent_checkpoint_id"],
                    }
                }
            
            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": data["checkpoint_id"],
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
            )
            count += 1
    
    @contextmanager
    def _cursor(self):
        """Context manager for cursor (not used for JSON, but required by interface)."""
        yield None
    
    def get_latest_checkpoint_info(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get info about the latest checkpoint for a thread.
        
        This is a convenience method for the resume CLI to show
        what state is available to resume from.
        
        Args:
            thread_id: The thread ID to look up
            
        Returns:
            Dict with checkpoint info or None if no checkpoints exist
        """
        checkpoints = self._get_thread_checkpoints(thread_id)
        if not checkpoints:
            return None
        
        filepath = checkpoints[0]
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Also try to get pending questions from the checkpoint
        checkpoint = self.serde.loads_typed(
            (data["checkpoint_type"], base64.b64decode(data["checkpoint_data"]))
        )
        channel_values = checkpoint.get("channel_values", {})
        
        return {
            "checkpoint_id": data["checkpoint_id"],
            "saved_at": data["saved_at"],
            "filepath": str(filepath),
            "pending_user_questions": channel_values.get("pending_user_questions", []),
            "ask_user_trigger": channel_values.get("ask_user_trigger"),
        }

