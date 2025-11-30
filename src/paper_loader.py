"""
Paper Input Loader for Paper Reproduction System

This module handles loading and validating paper inputs including
text, figures, and optional digitized data.
"""

from typing import TypedDict, Optional, List, Dict, Any
from typing_extensions import NotRequired
from pathlib import Path
import json


# ═══════════════════════════════════════════════════════════════════════
# Paper Input Schema
# ═══════════════════════════════════════════════════════════════════════

class FigureInput(TypedDict):
    """A figure from the paper for visual comparison."""
    id: str  # e.g., "Fig3a", "Fig3b"
    description: str  # What the figure shows
    image_path: str  # Path to figure image file
    digitized_data_path: NotRequired[str]  # Optional path to digitized CSV data


class PaperInput(TypedDict):
    """
    Complete input specification for a paper reproduction.
    
    This schema defines the expected format for paper inputs to the system.
    The system uses multimodal LLMs (GPT-4o, Claude) to analyze both text
    and figure images for comparison.
    """
    paper_id: str  # Unique identifier (e.g., "aluminum_nanoantenna_2013")
    paper_title: str  # Full paper title
    paper_text: str  # Extracted text from PDF (main text + methods + captions)
    paper_domain: NotRequired[str]  # Optional: plasmonics | photonic_crystal | etc.
    figures: List[FigureInput]  # Figures to reproduce with image paths
    supplementary_text: NotRequired[str]  # Optional supplementary material text


# ═══════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════

class ValidationError(Exception):
    """Raised when paper input validation fails."""
    pass


def validate_paper_input(paper_input: Dict[str, Any]) -> List[str]:
    """
    Validate a paper input dictionary.
    
    Args:
        paper_input: Dictionary containing paper input data
        
    Returns:
        List of validation warnings (empty if fully valid)
        
    Raises:
        ValidationError: If required fields are missing or invalid
    """
    errors = []
    warnings = []
    
    # Check required fields
    required_fields = ["paper_id", "paper_title", "paper_text", "figures"]
    for field in required_fields:
        if field not in paper_input:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        raise ValidationError(f"Paper input validation failed:\n" + "\n".join(errors))
    
    # Validate paper_id format
    paper_id = paper_input["paper_id"]
    if not paper_id or not isinstance(paper_id, str):
        raise ValidationError("paper_id must be a non-empty string")
    if " " in paper_id:
        warnings.append(f"paper_id contains spaces: '{paper_id}' - consider using underscores")
    
    # Validate paper_text is not empty
    paper_text = paper_input["paper_text"]
    if not paper_text or len(paper_text.strip()) < 100:
        raise ValidationError("paper_text is empty or too short (< 100 chars)")
    
    # Validate figures
    figures = paper_input["figures"]
    if not isinstance(figures, list):
        raise ValidationError("figures must be a list")
    
    if len(figures) == 0:
        warnings.append("No figures provided - system needs figure images for visual comparison")
    
    for i, fig in enumerate(figures):
        # Check required figure fields
        if "id" not in fig:
            errors.append(f"Figure {i}: missing 'id' field")
        if "image_path" not in fig:
            errors.append(f"Figure {i} ({fig.get('id', 'unknown')}): missing 'image_path' field")
        
        # Check if image file exists
        if "image_path" in fig:
            img_path = Path(fig["image_path"])
            if not img_path.exists():
                warnings.append(f"Figure {fig.get('id', i)}: image file not found: {fig['image_path']}")
            elif not img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                warnings.append(f"Figure {fig.get('id', i)}: unusual image format: {img_path.suffix}")
        
        # Check digitized data if provided
        if fig.get("digitized_data_path"):
            data_path = Path(fig["digitized_data_path"])
            if not data_path.exists():
                warnings.append(f"Figure {fig.get('id', i)}: digitized data file not found: {fig['digitized_data_path']}")
    
    if errors:
        raise ValidationError(f"Paper input validation failed:\n" + "\n".join(errors))
    
    return warnings


def validate_domain(domain: str) -> bool:
    """Check if domain is one of the recognized values."""
    valid_domains = [
        "plasmonics",
        "photonic_crystal", 
        "metamaterial",
        "thin_film",
        "waveguide",
        "strong_coupling",
        "nonlinear",
        "other"
    ]
    return domain in valid_domains


# ═══════════════════════════════════════════════════════════════════════
# Loading Functions
# ═══════════════════════════════════════════════════════════════════════

def load_paper_input(json_path: str) -> PaperInput:
    """
    Load and validate a paper input from a JSON file.
    
    Args:
        json_path: Path to JSON file containing paper input
        
    Returns:
        Validated PaperInput dictionary
        
    Raises:
        ValidationError: If validation fails
        FileNotFoundError: If JSON file not found
        json.JSONDecodeError: If JSON is malformed
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Paper input file not found: {json_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    warnings = validate_paper_input(data)
    
    if warnings:
        print(f"Paper input loaded with {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  ⚠️  {w}")
    
    return data


def create_paper_input(
    paper_id: str,
    paper_title: str,
    paper_text: str,
    figures: List[Dict[str, str]],
    paper_domain: str = "other",
    supplementary_text: Optional[str] = None
) -> PaperInput:
    """
    Create a PaperInput programmatically.
    
    Args:
        paper_id: Unique identifier for the paper
        paper_title: Full paper title
        paper_text: Extracted text from paper PDF
        figures: List of figure dicts with id, description, image_path
        paper_domain: Domain classification (default: "other")
        supplementary_text: Optional supplementary material text
        
    Returns:
        Validated PaperInput dictionary
    """
    paper_input = {
        "paper_id": paper_id,
        "paper_title": paper_title,
        "paper_text": paper_text,
        "paper_domain": paper_domain,
        "figures": figures,
    }
    
    if supplementary_text:
        paper_input["supplementary_text"] = supplementary_text
    
    validate_paper_input(paper_input)
    
    return paper_input


def get_figure_by_id(paper_input: PaperInput, figure_id: str) -> Optional[FigureInput]:
    """
    Get a specific figure from paper input by its ID.
    
    Args:
        paper_input: The paper input dictionary
        figure_id: The figure ID to find (e.g., "Fig3a")
        
    Returns:
        FigureInput dict if found, None otherwise
    """
    for fig in paper_input.get("figures", []):
        if fig.get("id") == figure_id:
            return fig
    return None


def list_figure_ids(paper_input: PaperInput) -> List[str]:
    """Get list of all figure IDs in paper input."""
    return [fig.get("id", f"unknown_{i}") for i, fig in enumerate(paper_input.get("figures", []))]


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
    "supplementary_text": "[Optional supplementary material text]"
}


if __name__ == "__main__":
    # Example usage
    print("Example paper input structure:")
    print(json.dumps(EXAMPLE_PAPER_INPUT, indent=2))
    
    print("\nValidating example...")
    try:
        # This will generate warnings since paths don't exist
        warnings = validate_paper_input(EXAMPLE_PAPER_INPUT)
        print(f"Validation passed with {len(warnings)} warnings")
    except ValidationError as e:
        print(f"Validation failed: {e}")

