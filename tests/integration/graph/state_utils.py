"""Utility helpers for graph integration state setup."""

from __future__ import annotations

from typing import Any

from schemas.state import ReproState, create_initial_state

DEFAULT_PAPER_ID = "graph_integration_test"
DEFAULT_PAPER_TEXT = "Graph integration test content."


def build_state(
    *,
    paper_id: str = DEFAULT_PAPER_ID,
    paper_text: str = DEFAULT_PAPER_TEXT,
    **overrides: Any,
) -> ReproState:
    """Create a fresh repro state with optional overrides applied."""
    state = create_initial_state(paper_id=paper_id, paper_text=paper_text)
    state.update(overrides)
    return state


def apply_state_overrides(state: ReproState, **overrides: Any) -> ReproState:
    """Apply arbitrary overrides to an existing state dict and return it."""
    state.update(overrides)
    return state


