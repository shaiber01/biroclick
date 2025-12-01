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


class DataFileInput(TypedDict):
    """A data file from the supplementary materials."""
    id: str  # e.g., "S1_spectrum", "geometry_params"
    description: str  # What the data file contains
    file_path: str  # Path to the data file (CSV, Excel, JSON, etc.)
    data_type: str  # Type hint: "spectrum", "geometry", "parameters", "time_series", "other"


class SupplementaryInput(TypedDict, total=False):
    """
    Structured supplementary materials from the paper.
    
    Scientific papers often have critical information in supplementary materials:
    - Additional methods details
    - Extended data tables
    - Supplementary figures
    - Raw data files
    """
    supplementary_text: str  # Extracted text from supplementary PDF
    supplementary_figures: List[FigureInput]  # Supplementary figure images
    supplementary_data_files: List[DataFileInput]  # CSV, Excel, etc. data files


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
    supplementary: NotRequired[SupplementaryInput]  # Structured supplementary materials


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


def validate_figure_image(image_path: str) -> List[str]:
    """
    Check if a figure image is suitable for vision models.
    
    Vision models (GPT-4o, Claude) work best with certain image characteristics.
    This function checks for common issues that may affect comparison quality.
    
    Args:
        image_path: Path to the figure image file
        
    Returns:
        List of warnings (empty if image is suitable)
    """
    warnings = []
    path = Path(image_path)
    
    if not path.exists():
        warnings.append(f"Image file not found: {image_path}")
        return warnings
    
    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 5:
        warnings.append(f"Large file size ({size_mb:.1f}MB) - consider compressing to save tokens")
    
    # Try to check image dimensions using PIL if available
    try:
        from PIL import Image
        
        img = Image.open(path)
        width, height = img.size
        
        if width < 512 or height < 512:
            warnings.append(
                f"Low resolution ({width}x{height}) - vision models work best with ≥512px on each side"
            )
        
        if width > 4096 or height > 4096:
            warnings.append(
                f"Very high resolution ({width}x{height}) - consider resizing to ≤4096px to save tokens"
            )
        
        # Check for very unusual aspect ratios
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 5:
            warnings.append(
                f"Extreme aspect ratio ({aspect_ratio:.1f}:1) - may be cropped by vision models"
            )
        
        img.close()
        
    except ImportError:
        # PIL not available - skip dimension checks
        warnings.append(
            "PIL/Pillow not installed - cannot check image dimensions. "
            "Install with: pip install Pillow"
        )
    except Exception as e:
        warnings.append(f"Could not analyze image: {e}")
    
    return warnings


# ═══════════════════════════════════════════════════════════════════════
# Token Cost Estimation
# ═══════════════════════════════════════════════════════════════════════

def estimate_token_cost(paper_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimate token usage and cost for reproducing a paper.
    
    This provides a rough estimate based on:
    - Paper text length
    - Number of figures (vision models charge per image)
    - Expected number of agent calls
    
    NOTE: This is a ROUGH estimate. Actual costs depend on:
    - Number of revisions needed
    - Model used (GPT-4o, Claude, etc.)
    - Complexity of the reproduction
    
    Args:
        paper_input: Paper input dictionary
        
    Returns:
        Dictionary with token and cost estimates
    """
    # Text tokens (rough estimate: ~4 chars per token)
    text_tokens = len(paper_input.get("paper_text", "")) / 4
    
    # Supplementary text if present
    supplementary = paper_input.get("supplementary", {})
    if supplementary.get("supplementary_text"):
        text_tokens += len(supplementary["supplementary_text"]) / 4
    
    # Image tokens (OpenAI GPT-4o pricing as reference)
    # ~170 tokens per 512x512 tile, typical figure = ~4 tiles = 680 tokens
    TOKENS_PER_FIGURE = 680
    
    figures = paper_input.get("figures", [])
    supp_figures = supplementary.get("supplementary_figures", [])
    total_figures = len(figures) + len(supp_figures)
    image_tokens = total_figures * TOKENS_PER_FIGURE
    
    # Estimate agent calls by workflow phase
    # Planning phase: 2 calls with full paper text
    planner_input_tokens = 2 * text_tokens
    
    # Per stage estimates (assume 4 stages average)
    num_stages = max(4, len(figures))  # At least one stage per key figure
    
    # Each stage involves:
    # - Design: 1 call with ~30% of paper context
    # - CodeGen: 2 calls (initial + 1 revision average)
    # - Review: 2 calls
    # - Analysis: 1 call with images
    stage_text_fraction = 0.3  # Only relevant parts of paper needed
    per_stage_text = text_tokens * stage_text_fraction
    per_stage_input = (
        per_stage_text +  # Design
        per_stage_text * 2 +  # CodeGen (2 calls)
        per_stage_text * 2 +  # Review (2 calls)
        per_stage_text + image_tokens / num_stages  # Analysis with figure
    )
    
    total_stages_input = per_stage_input * num_stages
    
    # Supervisor: 1 call per stage decision
    supervisor_input = num_stages * (per_stage_text * 0.5)
    
    # Report generation: 1 call with summaries
    report_input = text_tokens * 0.2 + image_tokens
    
    # Total input tokens
    total_input_estimate = (
        planner_input_tokens +
        total_stages_input +
        supervisor_input +
        report_input
    )
    
    # Output tokens (typically 20-30% of input for this use case)
    total_output_estimate = total_input_estimate * 0.25
    
    # Cost calculation (GPT-4o pricing as of late 2024)
    # Input: $2.50 per 1M tokens
    # Output: $10.00 per 1M tokens
    INPUT_COST_PER_MILLION = 2.50
    OUTPUT_COST_PER_MILLION = 10.00
    
    input_cost = total_input_estimate * INPUT_COST_PER_MILLION / 1_000_000
    output_cost = total_output_estimate * OUTPUT_COST_PER_MILLION / 1_000_000
    total_cost = input_cost + output_cost
    
    return {
        "estimated_input_tokens": int(total_input_estimate),
        "estimated_output_tokens": int(total_output_estimate),
        "estimated_total_tokens": int(total_input_estimate + total_output_estimate),
        "estimated_cost_usd": round(total_cost, 2),
        "cost_breakdown": {
            "input_cost_usd": round(input_cost, 2),
            "output_cost_usd": round(output_cost, 2),
        },
        "assumptions": {
            "num_figures": total_figures,
            "num_stages_estimated": num_stages,
            "text_chars": len(paper_input.get("paper_text", "")),
            "model_pricing": "GPT-4o (late 2024)",
        },
        "warning": (
            "This is a rough estimate. Actual costs depend on number of revisions, "
            "model selection, and reproduction complexity. "
            "Budget 2-3x this estimate for complex papers."
        )
    }


def load_paper_text(text_path: str) -> str:
    """
    Load paper text from a file (markdown, txt, etc.)
    
    Args:
        text_path: Path to text file
        
    Returns:
        Paper text as string
    """
    path = Path(text_path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


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
    supplementary_text: Optional[str] = None,
    supplementary_figures: Optional[List[Dict[str, str]]] = None,
    supplementary_data_files: Optional[List[Dict[str, str]]] = None
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
        supplementary_figures: Optional list of supplementary figure dicts
        supplementary_data_files: Optional list of data file dicts with id, description, file_path, data_type
        
    Returns:
        Validated PaperInput dictionary
        
    Example:
        paper_input = create_paper_input(
            paper_id="aluminum_nanoantenna_2013",
            paper_title="Aluminum nanoantenna complexes...",
            paper_text="... extracted text ...",
            figures=[
                {"id": "Fig3a", "description": "Transmission spectra", "image_path": "papers/fig3a.png"},
            ],
            paper_domain="plasmonics",
            supplementary_text="... supplementary methods ...",
            supplementary_figures=[
                {"id": "S1", "description": "Extended data", "image_path": "papers/figS1.png"},
            ],
            supplementary_data_files=[
                {"id": "S_data1", "description": "Optical constants", "file_path": "papers/optical_data.csv", "data_type": "spectrum"},
            ]
        )
    """
    paper_input = {
        "paper_id": paper_id,
        "paper_title": paper_title,
        "paper_text": paper_text,
        "paper_domain": paper_domain,
        "figures": figures,
    }
    
    # Build supplementary section if any supplementary content provided
    if supplementary_text or supplementary_figures or supplementary_data_files:
        supplementary: Dict[str, Any] = {}
        
        if supplementary_text:
            supplementary["supplementary_text"] = supplementary_text
        if supplementary_figures:
            supplementary["supplementary_figures"] = supplementary_figures
        if supplementary_data_files:
            supplementary["supplementary_data_files"] = supplementary_data_files
        
        paper_input["supplementary"] = supplementary
    
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


def get_supplementary_text(paper_input: PaperInput) -> Optional[str]:
    """Get supplementary text if available."""
    supplementary = paper_input.get("supplementary", {})
    return supplementary.get("supplementary_text")


def get_supplementary_figures(paper_input: PaperInput) -> List[FigureInput]:
    """Get list of supplementary figures."""
    supplementary = paper_input.get("supplementary", {})
    return supplementary.get("supplementary_figures", [])


def get_supplementary_data_files(paper_input: PaperInput) -> List[DataFileInput]:
    """Get list of supplementary data files."""
    supplementary = paper_input.get("supplementary", {})
    return supplementary.get("supplementary_data_files", [])


def get_data_file_by_type(paper_input: PaperInput, data_type: str) -> List[DataFileInput]:
    """
    Get supplementary data files by type.
    
    Args:
        paper_input: The paper input dictionary
        data_type: Type to filter by ("spectrum", "geometry", "parameters", etc.)
        
    Returns:
        List of DataFileInput dicts matching the type
    """
    data_files = get_supplementary_data_files(paper_input)
    return [f for f in data_files if f.get("data_type") == data_type]


def get_all_figures(paper_input: PaperInput) -> List[FigureInput]:
    """
    Get all figures including both main and supplementary figures.
    
    Returns:
        Combined list of main figures and supplementary figures
    """
    main_figures = paper_input.get("figures", [])
    supplementary_figures = get_supplementary_figures(paper_input)
    return main_figures + supplementary_figures


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



