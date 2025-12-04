"""Shared factories for building plans, progress structures, and states."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional


def make_stage(
    stage_id: str,
    stage_type: str,
    targets: Optional[Iterable[Any]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Return a stage dictionary with common defaults."""
    stage = {
        "stage_id": stage_id,
        "stage_type": stage_type,
        "targets": list(targets or []),
        "dependencies": overrides.pop("dependencies", []),
        "status": overrides.pop("status", "not_started"),
    }
    stage.update(overrides)
    return stage


def make_plan(stages: Optional[List[Dict[str, Any]]] = None, **overrides: Any) -> Dict[str, Any]:
    """Build a minimal plan structure."""
    plan = {
        "paper_id": overrides.pop("paper_id", "test_plan"),
        "title": overrides.pop("title", "Test Plan"),
        "stages": stages or [],
        "targets": overrides.pop("targets", []),
    }
    plan.update(overrides)
    return plan


def make_progress(
    stages: Optional[List[Dict[str, Any]]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a progress structure mirroring the plan."""
    progress = {"stages": stages or []}
    progress.update(overrides)
    return progress


def augment_state(state: Dict[str, Any], **updates: Any) -> Dict[str, Any]:
    """Return a copy of state updated with the provided key/value pairs."""
    next_state = deepcopy(state)
    next_state.update(updates)
    return next_state



