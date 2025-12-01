"""
Paper Input Loader for Paper Reproduction System

This module handles loading and validating paper inputs including
text, figures, and optional digitized data.

Supports two loading modes:
1. JSON-based: Load from a pre-prepared JSON file with explicit figure paths
2. Markdown-based: Parse markdown (from marker/nougat), extract and download figures
"""

from typing import TypedDict, Optional, List, Dict, Any
from typing_extensions import NotRequired
from pathlib import Path
import json
import re
import shutil
import urllib.request
import urllib.error
from urllib.parse import urlparse, unquote, urljoin


# ═══════════════════════════════════════════════════════════════════════
# Supported Image Formats
# ═══════════════════════════════════════════════════════════════════════

# Image formats supported by vision models (GPT-4o, Claude)
SUPPORTED_IMAGE_FORMATS = {
    # Raster formats - widely supported
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    # Additional raster formats
    '.bmp': 'image/bmp',
    '.tiff': 'image/tiff',
    '.tif': 'image/tiff',
    '.ico': 'image/x-icon',
    # Vector formats (may need conversion for some models)
    '.svg': 'image/svg+xml',
    # Scientific formats (require special handling)
    '.eps': 'application/postscript',
    '.pdf': 'application/pdf',
}

# Formats that vision models handle well
VISION_MODEL_PREFERRED_FORMATS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}


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


# ═══════════════════════════════════════════════════════════════════════
# Markdown Paper Loading with Figure Download
# ═══════════════════════════════════════════════════════════════════════

class FigureDownloadError(Exception):
    """Raised when a figure cannot be downloaded."""
    pass


def extract_figures_from_markdown(markdown_text: str) -> List[Dict[str, str]]:
    """
    Extract figure references from markdown text.
    
    Supports:
    - Markdown images: ![alt text](url)
    - Markdown images with title: ![alt text](url "title")
    - HTML img tags: <img src="url" alt="..." />
    
    Args:
        markdown_text: The markdown content
        
    Returns:
        List of dicts with 'alt', 'url', and 'original_match' keys
    """
    figures = []
    
    # Pattern for markdown images: ![alt](url) or ![alt](url "title")
    markdown_pattern = r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)'
    for match in re.finditer(markdown_pattern, markdown_text):
        alt_text = match.group(1).strip()
        url = match.group(2).strip()
        figures.append({
            'alt': alt_text,
            'url': url,
            'original_match': match.group(0)
        })
    
    # Pattern for HTML img tags: <img src="url" ... />
    # Handles both src before alt and alt before src
    html_pattern = r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*/?>'
    alt_pattern = r'alt=["\']([^"\']*)["\']'
    
    for match in re.finditer(html_pattern, markdown_text, re.IGNORECASE):
        url = match.group(1).strip()
        # Try to find alt text in the same tag
        alt_match = re.search(alt_pattern, match.group(0), re.IGNORECASE)
        alt_text = alt_match.group(1).strip() if alt_match else ''
        
        # Avoid duplicates if same URL found in both formats
        if not any(f['url'] == url for f in figures):
            figures.append({
                'alt': alt_text,
                'url': url,
                'original_match': match.group(0)
            })
    
    return figures


def resolve_figure_url(url: str, base_path: Optional[Path] = None, base_url: Optional[str] = None) -> str:
    """
    Resolve a figure URL, handling relative paths.
    
    Resolution order:
    1. If URL is absolute (http/https/file://), use as-is
    2. If base_url is provided and URL is relative, join with base_url
    3. If base_path is provided and URL is relative, resolve against base_path
    4. Otherwise, return URL as-is
    
    Args:
        url: The figure URL or path from markdown
        base_path: Optional base path (typically the markdown file's directory)
        base_url: Optional base URL for remote relative paths
        
    Returns:
        Resolved URL or path string
    """
    parsed = urlparse(url)
    
    # Already absolute URL
    if parsed.scheme in ('http', 'https', 'file'):
        return url
    
    # Relative URL with base_url provided
    if base_url and not parsed.scheme:
        return urljoin(base_url, url)
    
    # Relative path with base_path provided
    if base_path and not parsed.scheme:
        resolved = base_path / url
        return str(resolved)
    
    # Return as-is
    return url


def generate_figure_id(index: int, alt_text: str, url: str) -> str:
    """
    Generate a figure ID from available information.
    
    Extraction priority:
    1. Figure number from alt text (e.g., "Figure 3a" -> "Fig3a")
    2. Figure number from URL filename
    3. Sequential numbering as fallback
    
    Args:
        index: Figure index (0-based)
        alt_text: Alt text from markdown
        url: Original URL
        
    Returns:
        A figure ID string like "Fig1" or "Fig3a"
    """
    # Try to extract figure number from alt text
    fig_match = re.search(r'(?:fig(?:ure)?|fig\.?)\s*(\d+[a-z]?)', alt_text, re.IGNORECASE)
    if fig_match:
        return f"Fig{fig_match.group(1)}"
    
    # Try to extract from URL filename
    parsed = urlparse(url)
    filename = Path(unquote(parsed.path)).stem
    fig_match = re.search(r'fig(?:ure)?[_\-]?(\d+[a-z]?)', filename, re.IGNORECASE)
    if fig_match:
        return f"Fig{fig_match.group(1)}"
    
    # Fall back to sequential numbering
    return f"Fig{index + 1}"


def get_file_extension(url: str, default: str = '.png') -> str:
    """
    Determine the file extension from a URL.
    
    Args:
        url: The figure URL
        default: Default extension if none can be determined
        
    Returns:
        File extension including the dot (e.g., '.png')
    """
    parsed = urlparse(url)
    path = unquote(parsed.path)
    ext = Path(path).suffix.lower()
    
    if ext in SUPPORTED_IMAGE_FORMATS:
        return ext
    return default


def download_figure(
    url: str, 
    output_path: Path, 
    timeout: int = 30,
    base_path: Optional[Path] = None
) -> None:
    """
    Download a figure from a URL or copy from local path.
    
    Supports:
    - Remote URLs (http, https)
    - Local file paths (absolute or relative)
    - file:// URLs
    
    Args:
        url: URL or path of the figure to download
        output_path: Local path to save the figure
        timeout: Download timeout in seconds (for remote URLs)
        base_path: Base path for resolving relative local paths
        
    Raises:
        FigureDownloadError: If download/copy fails
    """
    try:
        parsed = urlparse(url)
        
        if parsed.scheme in ('http', 'https'):
            # Remote URL - download it
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; PaperLoader/1.0)'}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = response.read()
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(content)
                
        elif parsed.scheme == 'file':
            # file:// URL - extract path and copy
            local_path = Path(unquote(parsed.path))
            if not local_path.exists():
                raise FigureDownloadError(f"Local file not found: {local_path}")
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, output_path)
            
        else:
            # Assume local file path (relative or absolute)
            local_path = Path(url)
            
            # If relative and base_path provided, resolve against it
            if not local_path.is_absolute() and base_path:
                local_path = base_path / url
            
            if not local_path.exists():
                raise FigureDownloadError(f"Local file not found: {local_path}")
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, output_path)
            
    except urllib.error.URLError as e:
        raise FigureDownloadError(f"Failed to download {url}: {e}")
    except urllib.error.HTTPError as e:
        raise FigureDownloadError(f"HTTP error downloading {url}: {e.code} {e.reason}")
    except OSError as e:
        raise FigureDownloadError(f"Failed to save figure to {output_path}: {e}")


def extract_paper_title(markdown_text: str) -> str:
    """
    Extract paper title from markdown.
    
    Looks for (in order):
    1. First H1 heading: # Title
    2. HTML h1 tag: <h1>Title</h1>
    3. First non-empty, non-image line
    
    Args:
        markdown_text: The markdown content
        
    Returns:
        Paper title string, or "Untitled Paper" if not found
    """
    # Look for first H1 heading: # Title
    h1_match = re.search(r'^#\s+(.+?)(?:\n|$)', markdown_text, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()
    
    # Look for HTML h1
    html_h1_match = re.search(r'<h1[^>]*>(.+?)</h1>', markdown_text, re.IGNORECASE | re.DOTALL)
    if html_h1_match:
        # Strip any HTML tags inside
        title = re.sub(r'<[^>]+>', '', html_h1_match.group(1))
        return title.strip()
    
    # Look for first non-empty line as fallback
    for line in markdown_text.split('\n'):
        line = line.strip()
        if line and not line.startswith('!') and not line.startswith('<'):
            return line[:200]  # Truncate very long lines
    
    return "Untitled Paper"


# ═══════════════════════════════════════════════════════════════════════
# Paper Length Thresholds
# ═══════════════════════════════════════════════════════════════════════

# Character thresholds for paper length warnings
PAPER_LENGTH_NORMAL = 50_000       # ~12,500 tokens - typical paper
PAPER_LENGTH_LONG = 150_000        # ~37,500 tokens - long paper, consider trimming
PAPER_LENGTH_VERY_LONG = 300_000   # ~75,000 tokens - very long, may hit context limits


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English text)."""
    return len(text) // 4


def check_paper_length(text: str, label: str = "Paper") -> List[str]:
    """
    Check paper text length and return warnings if too long.
    
    Args:
        text: The paper text to check
        label: Label for the text (e.g., "Paper", "Supplementary")
        
    Returns:
        List of warning strings (empty if length is normal)
    """
    warnings = []
    char_count = len(text)
    token_estimate = estimate_tokens(text)
    
    if char_count > PAPER_LENGTH_VERY_LONG:
        warnings.append(
            f"{label} is VERY LONG: {char_count:,} chars (~{token_estimate:,} tokens). "
            f"This may exceed LLM context limits and significantly increase costs. "
            f"Consider removing references, acknowledgments, or non-essential sections."
        )
    elif char_count > PAPER_LENGTH_LONG:
        warnings.append(
            f"{label} is long: {char_count:,} chars (~{token_estimate:,} tokens). "
            f"Consider trimming references section to reduce costs."
        )
    
    return warnings


def load_paper_from_markdown(
    markdown_path: str,
    output_dir: str,
    paper_id: Optional[str] = None,
    paper_domain: str = "other",
    base_url: Optional[str] = None,
    supplementary_markdown_path: Optional[str] = None,
    supplementary_base_url: Optional[str] = None,
    download_figures: bool = True,
    figure_timeout: int = 30
) -> PaperInput:
    """
    Load a paper from a markdown file and download embedded figures.
    
    This function parses a markdown file (e.g., output from marker or nougat),
    extracts figure references, downloads them, and creates a PaperInput structure.
    
    Figure URL Resolution:
    - Absolute URLs (http/https): Downloaded directly
    - Relative URLs with base_url: Joined with base_url before downloading
    - Relative paths: Resolved against the markdown file's directory
    - file:// URLs: Extracted and copied
    
    Supported Image Formats:
    - Raster: PNG, JPG/JPEG, GIF, WebP, BMP, TIFF/TIF, ICO
    - Vector: SVG (may need conversion for some vision models)
    - Scientific: EPS, PDF (may need conversion)
    
    Paper Length Guidelines:
    - < 50K chars: Normal (most papers)
    - 50-150K chars: Long - consider trimming references
    - > 150K chars: Very long - may exceed context limits, trim non-essential sections
    
    Args:
        markdown_path: Path to the markdown file containing the paper
        output_dir: Directory to store downloaded figures
        paper_id: Optional paper ID (defaults to markdown filename)
        paper_domain: Domain classification for the paper
        base_url: Optional base URL for resolving relative URLs in markdown
        supplementary_markdown_path: Optional path to supplementary materials markdown
        supplementary_base_url: Optional base URL for supplementary figure URLs
        download_figures: Whether to download figures (if False, just extract info)
        figure_timeout: Timeout for figure downloads in seconds
        
    Returns:
        PaperInput dictionary with paper content and figure references
        
    Raises:
        FileNotFoundError: If markdown file doesn't exist
        ValidationError: If resulting paper input is invalid
        
    Example:
        # Load from local markdown with relative image paths
        paper_input = load_paper_from_markdown(
            markdown_path="papers/smith2023/paper.md",
            output_dir="papers/smith2023/figures",
            paper_id="smith2023_plasmon",
            paper_domain="plasmonics"
        )
        
        # Load with supplementary materials
        paper_input = load_paper_from_markdown(
            markdown_path="papers/smith2023/paper.md",
            output_dir="papers/smith2023/figures",
            supplementary_markdown_path="papers/smith2023/supplementary.md"
        )
        
        # Load from markdown with remote image URLs needing a base URL
        paper_input = load_paper_from_markdown(
            markdown_path="papers/downloaded.md",
            output_dir="papers/figures",
            base_url="https://arxiv.org/html/paper123/"
        )
    """
    md_path = Path(markdown_path)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")
    
    # Read markdown content
    with open(md_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()
    
    # Check paper length and collect warnings
    length_warnings: List[str] = []
    length_warnings.extend(check_paper_length(markdown_text, "Main paper"))
    
    # Generate paper_id from filename if not provided
    if paper_id is None:
        paper_id = md_path.stem.replace(' ', '_').replace('-', '_').lower()
    
    # Extract paper title
    paper_title = extract_paper_title(markdown_text)
    
    # Extract figure references
    figure_refs = extract_figures_from_markdown(markdown_text)
    
    # Set up output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Base path for resolving relative local paths
    md_base_path = md_path.parent
    
    # Process main paper figures
    figures: List[Dict[str, Any]] = []
    download_errors: List[str] = []
    
    print(f"Processing {len(figure_refs)} main figure(s)...")
    
    for i, fig_ref in enumerate(figure_refs):
        fig_id = generate_figure_id(i, fig_ref['alt'], fig_ref['url'])
        
        # Ensure unique IDs
        base_id = fig_id
        counter = 1
        while any(f['id'] == fig_id for f in figures):
            fig_id = f"{base_id}_{counter}"
            counter += 1
        
        # Resolve the URL (handle relative paths)
        resolved_url = resolve_figure_url(
            fig_ref['url'], 
            base_path=md_base_path, 
            base_url=base_url
        )
        
        # Determine output filename
        ext = get_file_extension(fig_ref['url'])
        fig_filename = f"{fig_id}{ext}"
        fig_output_path = output_path / fig_filename
        
        figure_entry: Dict[str, Any] = {
            'id': fig_id,
            'description': fig_ref['alt'] or f"Figure from paper",
            'image_path': str(fig_output_path),
            'source_url': fig_ref['url'],  # Keep original URL for reference
        }
        
        # Check if format is preferred by vision models
        if ext not in VISION_MODEL_PREFERRED_FORMATS:
            figure_entry['format_warning'] = (
                f"Format {ext} may need conversion for optimal vision model performance. "
                f"Preferred formats: {', '.join(sorted(VISION_MODEL_PREFERRED_FORMATS))}"
            )
        
        if download_figures:
            try:
                download_figure(
                    resolved_url, 
                    fig_output_path, 
                    timeout=figure_timeout,
                    base_path=md_base_path
                )
                print(f"  ✓ Downloaded {fig_id}: {fig_ref['url'][:60]}{'...' if len(fig_ref['url']) > 60 else ''}")
            except FigureDownloadError as e:
                download_errors.append(f"{fig_id}: {e}")
                print(f"  ✗ Failed to download {fig_id}: {e}")
                figure_entry['download_error'] = str(e)
        
        figures.append(figure_entry)
    
    # Process supplementary materials if provided
    supplementary_text: Optional[str] = None
    supplementary_figures: List[Dict[str, Any]] = []
    supp_download_errors: List[str] = []
    
    if supplementary_markdown_path:
        supp_path = Path(supplementary_markdown_path)
        if not supp_path.exists():
            print(f"\n⚠️  Supplementary file not found: {supplementary_markdown_path}")
        else:
            # Read supplementary markdown
            with open(supp_path, 'r', encoding='utf-8') as f:
                supplementary_text = f.read()
            
            # Check supplementary length
            length_warnings.extend(check_paper_length(supplementary_text, "Supplementary"))
            
            # Extract supplementary figures
            supp_figure_refs = extract_figures_from_markdown(supplementary_text)
            supp_base_path = supp_path.parent
            supp_base = supplementary_base_url or base_url  # Fall back to main base_url
            
            print(f"\nProcessing {len(supp_figure_refs)} supplementary figure(s)...")
            
            for i, fig_ref in enumerate(supp_figure_refs):
                # Generate ID with "S" prefix for supplementary
                fig_id = generate_figure_id(i, fig_ref['alt'], fig_ref['url'])
                if not fig_id.upper().startswith('S'):
                    fig_id = f"S{fig_id}"
                
                # Ensure unique IDs (check against both main and supp figures)
                all_existing_ids = [f['id'] for f in figures] + [f['id'] for f in supplementary_figures]
                base_id = fig_id
                counter = 1
                while fig_id in all_existing_ids:
                    fig_id = f"{base_id}_{counter}"
                    counter += 1
                
                # Resolve URL
                resolved_url = resolve_figure_url(
                    fig_ref['url'],
                    base_path=supp_base_path,
                    base_url=supp_base
                )
                
                # Determine output filename
                ext = get_file_extension(fig_ref['url'])
                fig_filename = f"{fig_id}{ext}"
                fig_output_path = output_path / fig_filename
                
                figure_entry = {
                    'id': fig_id,
                    'description': fig_ref['alt'] or "Supplementary figure",
                    'image_path': str(fig_output_path),
                    'source_url': fig_ref['url'],
                }
                
                if ext not in VISION_MODEL_PREFERRED_FORMATS:
                    figure_entry['format_warning'] = (
                        f"Format {ext} may need conversion for optimal vision model performance."
                    )
                
                if download_figures:
                    try:
                        download_figure(
                            resolved_url,
                            fig_output_path,
                            timeout=figure_timeout,
                            base_path=supp_base_path
                        )
                        print(f"  ✓ Downloaded {fig_id}: {fig_ref['url'][:60]}{'...' if len(fig_ref['url']) > 60 else ''}")
                    except FigureDownloadError as e:
                        supp_download_errors.append(f"{fig_id}: {e}")
                        print(f"  ✗ Failed to download {fig_id}: {e}")
                        figure_entry['download_error'] = str(e)
                
                supplementary_figures.append(figure_entry)
    
    # Build PaperInput
    paper_input: Dict[str, Any] = {
        'paper_id': paper_id,
        'paper_title': paper_title,
        'paper_text': markdown_text,
        'paper_domain': paper_domain,
        'figures': figures,
    }
    
    # Add supplementary section if we have any supplementary content
    if supplementary_text or supplementary_figures:
        paper_input['supplementary'] = {}
        if supplementary_text:
            paper_input['supplementary']['supplementary_text'] = supplementary_text
        if supplementary_figures:
            paper_input['supplementary']['supplementary_figures'] = supplementary_figures
    
    # Combine all download errors for reporting
    all_download_errors = download_errors + supp_download_errors
    
    # Calculate total text length and tokens
    total_text_length = len(markdown_text)
    if supplementary_text:
        total_text_length += len(supplementary_text)
    total_tokens = estimate_tokens(markdown_text)
    if supplementary_text:
        total_tokens += estimate_tokens(supplementary_text)
    
    # Report results
    print(f"\n{'='*60}")
    print(f"Paper loaded from markdown:")
    print(f"  Title: {paper_title[:80]}{'...' if len(paper_title) > 80 else ''}")
    print(f"  ID: {paper_id}")
    print(f"  Main text: {len(markdown_text):,} chars (~{estimate_tokens(markdown_text):,} tokens)")
    if supplementary_text:
        print(f"  Supplementary text: {len(supplementary_text):,} chars (~{estimate_tokens(supplementary_text):,} tokens)")
    print(f"  Total: {total_text_length:,} chars (~{total_tokens:,} tokens)")
    print(f"  Main figures: {len(figures)}")
    if supplementary_figures:
        print(f"  Supplementary figures: {len(supplementary_figures)}")
    if download_figures:
        total_figs = len(figures) + len(supplementary_figures)
        successful = total_figs - len(all_download_errors)
        print(f"  Figures downloaded: {successful}/{total_figs}")
    print(f"{'='*60}")
    
    # Display length warnings
    if length_warnings:
        print(f"\n📏 Length warnings:")
        for w in length_warnings:
            print(f"  ⚠️  {w}")
    
    if all_download_errors:
        print(f"\n⚠️  {len(all_download_errors)} figure(s) failed to download:")
        for err in all_download_errors[:5]:  # Show first 5 errors
            print(f"    - {err}")
        if len(all_download_errors) > 5:
            print(f"    ... and {len(all_download_errors) - 5} more")
    
    # Validate (will raise if critical errors)
    warnings = validate_paper_input(paper_input)
    if warnings:
        print(f"\nValidation warnings:")
        for w in warnings:
            print(f"  ⚠️  {w}")
    
    return paper_input


def save_paper_input_json(paper_input: PaperInput, output_path: str) -> None:
    """
    Save a PaperInput to a JSON file.
    
    Useful for saving the parsed paper for later use without re-downloading figures.
    
    Args:
        paper_input: The PaperInput dictionary
        output_path: Path for the output JSON file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(paper_input, f, indent=2, ensure_ascii=False)
    
    print(f"Paper input saved to: {output_path}")


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
    
    # Show markdown loading example
    print("\n" + "=" * 60)
    print("Markdown Loading Example")
    print("=" * 60)
    print("""
Usage:
    from src.paper_loader import load_paper_from_markdown
    
    # Basic: Load from local markdown with relative image paths
    paper_input = load_paper_from_markdown(
        markdown_path="papers/smith2023/paper.md",
        output_dir="papers/smith2023/figures",
        paper_id="smith2023_plasmon",
        paper_domain="plasmonics"
    )
    
    # With supplementary materials (separate SI PDF converted to markdown)
    paper_input = load_paper_from_markdown(
        markdown_path="papers/smith2023/paper.md",
        output_dir="papers/smith2023/figures",
        supplementary_markdown_path="papers/smith2023/supplementary.md",
        paper_id="smith2023_plasmon"
    )
    
    # With a base URL for remote relative images
    paper_input = load_paper_from_markdown(
        markdown_path="papers/downloaded.md",
        output_dir="papers/figures",
        base_url="https://example.com/papers/images/"
    )

Supported image formats: {}
Preferred for vision models: {}

Paper length thresholds:
    < {:,} chars: Normal (most papers)
    {:,}-{:,} chars: Long - consider trimming references
    > {:,} chars: Very long - may exceed context limits
""".format(
        ', '.join(sorted(SUPPORTED_IMAGE_FORMATS.keys())),
        ', '.join(sorted(VISION_MODEL_PREFERRED_FORMATS)),
        PAPER_LENGTH_NORMAL,
        PAPER_LENGTH_NORMAL,
        PAPER_LENGTH_LONG,
        PAPER_LENGTH_LONG
    ))



