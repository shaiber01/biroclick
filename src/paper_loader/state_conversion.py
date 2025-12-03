"""
State conversion for paper inputs.

This module handles converting PaperInput to ReproState for graph invocation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .validation import validate_paper_input


logger = logging.getLogger(__name__)


def create_state_from_paper_input(
    paper_input: Dict[str, Any],
    runtime_budget_minutes: float = 120.0,
    runtime_config: Optional[Dict[str, Any]] = None,
    hardware_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convert a PaperInput to initial ReproState with explicit field mapping.
    
    This function ensures proper data flow from PaperInput into ReproState,
    handling field name differences (e.g., 'figures' → 'paper_figures').
    
    Use this function instead of directly passing paper_input to app.invoke()
    to ensure all fields are properly mapped.
    
    Args:
        paper_input: Validated PaperInput dictionary
        runtime_budget_minutes: Total runtime budget in minutes (default: 120)
        runtime_config: Optional RuntimeConfig dict for timeouts, limits, etc.
        hardware_config: Optional HardwareConfig dict for CPU cores, RAM, etc.
        
    Returns:
        Initialized ReproState dictionary ready for graph invocation
        
    Example:
        from src.paper_loader import load_paper_from_markdown, create_state_from_paper_input
        
        # Load paper
        paper_input = load_paper_from_markdown(
            "papers/my_paper.md",
            paper_id="my_paper_2023"
        )
        
        # Convert to state
        initial_state = create_state_from_paper_input(paper_input)
        
        # Run the graph
        result = app.invoke(initial_state)
    """
    # Import here to avoid circular imports
    from schemas.state import create_initial_state, DEFAULT_RUNTIME_CONFIG, DEFAULT_HARDWARE_CONFIG
    
    # Validate input first
    warnings = validate_paper_input(paper_input)
    if warnings:
        for warning in warnings:
            logger.warning("Paper input warning: %s", warning)
    
    # Use provided configs or defaults
    rt_config = runtime_config if runtime_config is not None else DEFAULT_RUNTIME_CONFIG
    hw_config = hardware_config if hardware_config is not None else DEFAULT_HARDWARE_CONFIG
    
    # Create initial state with core fields
    state = create_initial_state(
        paper_id=paper_input["paper_id"],
        paper_text=paper_input["paper_text"],
        paper_domain=paper_input.get("paper_domain", "other"),
        runtime_budget_minutes=runtime_budget_minutes,
        runtime_config=rt_config,
        hardware_config=hw_config,
    )
    
    # Explicit field mapping: PaperInput → ReproState
    # This is the key part that ensures figures flow properly
    state["paper_title"] = paper_input.get("paper_title", "")
    
    # Map 'figures' → 'paper_figures' (different field names)
    state["paper_figures"] = paper_input.get("figures", [])
    
    # Handle supplementary materials if present
    supplementary = paper_input.get("supplementary")
    if supplementary:
        # Store raw supplementary for agents that need it
        # The state doesn't have a dedicated field, but we can add to assumptions later
        # For now, append supplementary text to paper_text context
        if supplementary.get("supplementary_text"):
            state["paper_text"] = (
                state["paper_text"] +
                "\n\n--- SUPPLEMENTARY MATERIALS ---\n\n" +
                supplementary["supplementary_text"]
            )
        
        # Supplementary figures get added to paper_figures
        supp_figures = supplementary.get("supplementary_figures", [])
        if supp_figures:
            for fig in supp_figures:
                # Mark supplementary figures with prefix for clarity
                fig_copy = dict(fig)
                if not fig_copy.get("id", "").startswith("S"):
                    fig_copy["id"] = f"S_{fig_copy.get('id', 'unknown')}"
                state["paper_figures"].append(fig_copy)
    
    return state


def load_paper_text(text_path: str) -> str:
    """
    Load paper text from a file (markdown, txt, etc.)
    
    Args:
        text_path: Path to text file
        
    Returns:
        Paper text as string
        
    Raises:
        FileNotFoundError: If text file doesn't exist
    """
    path = Path(text_path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()




