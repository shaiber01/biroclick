"""
Type definitions for paper input structures.

This module defines the TypedDicts that represent paper input data:
- FigureInput: A figure from the paper for visual comparison
- DataFileInput: A data file from supplementary materials
- SupplementaryInput: Structured supplementary materials
- PaperInput: Complete input specification for paper reproduction
"""

from typing import TypedDict, List
from typing_extensions import NotRequired


class FigureInput(TypedDict):
    """
    A figure from the paper for visual comparison.
    
    Attributes:
        id: Unique identifier, e.g., "Fig3a", "Fig3b"
        description: What the figure shows
        image_path: Path to figure image file
        digitized_data_path: Optional path to digitized CSV data
    """
    id: str
    description: str
    image_path: str
    digitized_data_path: NotRequired[str]


class DataFileInput(TypedDict):
    """
    A data file from the supplementary materials.
    
    Attributes:
        id: Unique identifier, e.g., "S1_spectrum", "geometry_params"
        description: What the data file contains
        file_path: Path to the data file (CSV, Excel, JSON, etc.)
        data_type: Type hint: "spectrum", "geometry", "parameters", "time_series", "other"
    """
    id: str
    description: str
    file_path: str
    data_type: str


class SupplementaryInput(TypedDict, total=False):
    """
    Structured supplementary materials from the paper.
    
    Scientific papers often have critical information in supplementary materials:
    - Additional methods details
    - Extended data tables
    - Supplementary figures
    - Raw data files
    
    Attributes:
        supplementary_text: Extracted text from supplementary PDF
        supplementary_figures: Supplementary figure images
        supplementary_data_files: CSV, Excel, etc. data files
    """
    supplementary_text: str
    supplementary_figures: List[FigureInput]
    supplementary_data_files: List[DataFileInput]


class PaperInput(TypedDict):
    """
    Complete input specification for a paper reproduction.
    
    This schema defines the expected format for paper inputs to the system.
    The system uses multimodal LLMs (Claude, GPT-4, etc.) to analyze both text
    and figure images for comparison.
    
    Attributes:
        paper_id: Unique identifier (e.g., "aluminum_nanoantenna_2013")
        paper_title: Full paper title
        paper_text: Extracted text from PDF (main text + methods + captions)
        paper_domain: Optional domain: plasmonics | photonic_crystal | etc.
        figures: Figures to reproduce with image paths
        supplementary: Structured supplementary materials
    """
    paper_id: str
    paper_title: str
    paper_text: str
    paper_domain: NotRequired[str]
    figures: List[FigureInput]
    supplementary: NotRequired[SupplementaryInput]


# ═══════════════════════════════════════════════════════════════════════
# Example Paper Input Structure
# ═══════════════════════════════════════════════════════════════════════

EXAMPLE_PAPER_INPUT = {
    "paper_id": "aluminum_nanoantenna_2013",
    "paper_title": "Aluminum nanoantenna complexes for strong coupling between J-aggregates and localized surface plasmons",
    "paper_text": """
    [Extracted paper text would go here - typically 10-50KB of text including
    abstract, introduction, methods, results, discussion, and figure captions]
    """,
    "paper_domain": "plasmonics",
    "figures": [
        {
            "id": "Fig2a",
            "description": "J-aggregate absorption spectrum showing exciton peak at ~590nm",
            "image_path": "papers/al_nanoantenna/figures/fig2a.png",
            "digitized_data_path": "papers/al_nanoantenna/data/fig2a_absorption.csv"
        },
        {
            "id": "Fig3a",
            "description": "Transmission spectra of bare Al nanodisks for different diameters",
            "image_path": "papers/al_nanoantenna/figures/fig3a.png"
        },
        {
            "id": "Fig3b",
            "description": "Transmission spectra of J-aggregate coated Al nanodisks showing Rabi splitting",
            "image_path": "papers/al_nanoantenna/figures/fig3b.png"
        },
        {
            "id": "Fig4",
            "description": "Dispersion diagram showing anticrossing behavior",
            "image_path": "papers/al_nanoantenna/figures/fig4.png"
        }
    ],
    # Supplementary materials use nested SupplementaryInput structure
    "supplementary": {
        "supplementary_text": "[Optional supplementary material text]",
        "supplementary_figures": [
            {
                "id": "FigS1",
                "description": "Extended material characterization",
                "image_path": "papers/al_nanoantenna/figures/figS1.png"
            }
        ],
        "supplementary_data_files": [
            {
                "id": "S_Al_nk",
                "description": "Aluminum optical constants used in simulations",
                "file_path": "papers/al_nanoantenna/data/Al_optical_constants.csv",
                "data_type": "spectrum"
            }
        ]
    }
}




