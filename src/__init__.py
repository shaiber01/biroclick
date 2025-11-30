"""
ReproLab: Multi-Agent Paper Reproduction System

A LangGraph-based system for automatically reproducing simulation results
from optics and metamaterials research papers using Meep FDTD simulations.
"""

__version__ = "0.1.0"
__author__ = "ReproLab Team"

from .graph import create_repro_graph
from ..schemas.state import ReproState, create_initial_state

__all__ = [
    "create_repro_graph",
    "ReproState", 
    "create_initial_state"
]

