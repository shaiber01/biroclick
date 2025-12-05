"""
Paper loading functions.

This module provides the main entry points for loading paper inputs:
- load_paper_input: Load from JSON file
- load_paper_from_markdown: Load from markdown with figure extraction
- create_paper_input: Create programmatically
- Accessor functions for paper figures and supplementary data
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .schemas import PaperInput, FigureInput, DataFileInput
from .config import VISION_MODEL_PREFERRED_FORMATS
from .validation import validate_paper_input
from .cost_estimation import estimate_tokens, check_paper_length
from .markdown_parser import (
    extract_figures_from_markdown,
    extract_paper_title,
    resolve_figure_url,
    generate_figure_id,
    get_file_extension,
)
from .downloader import download_figure, FigureDownloadError


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Internal Helper Functions
# ═══════════════════════════════════════════════════════════════════════

def _process_figure_refs(
    figure_refs: List[Dict[str, str]],
    output_path: Path,
    base_path: Path,
    base_url: Optional[str],
    download_figures: bool,
    figure_timeout: int,
    existing_ids: List[str],
    id_prefix: str = "",
    default_description: str = "Figure from paper"
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Process figure references and optionally download them.
    
    Args:
        figure_refs: List of figure reference dicts from markdown extraction
        output_path: Directory to save downloaded figures
        base_path: Base path for resolving relative URLs
        base_url: Optional base URL for remote relative paths
        download_figures: Whether to actually download the figures
        figure_timeout: Download timeout in seconds
        existing_ids: List of existing figure IDs to avoid duplicates
        id_prefix: Prefix to add to figure IDs (e.g., "S" for supplementary)
        default_description: Default description if alt text is empty
        
    Returns:
        Tuple of (figures list, download_errors list)
    """
    figures: List[Dict[str, Any]] = []
    download_errors: List[str] = []
    all_ids = list(existing_ids)
    
    for i, fig_ref in enumerate(figure_refs):
        # Generate figure ID
        fig_id = generate_figure_id(i, fig_ref['alt'], fig_ref['url'])
        
        # Add prefix if specified
        if id_prefix and not fig_id.upper().startswith(id_prefix.upper()):
            fig_id = f"{id_prefix}{fig_id}"
        
        # Ensure unique IDs
        base_id = fig_id
        counter = 1
        while fig_id in all_ids:
            fig_id = f"{base_id}_{counter}"
            counter += 1
        all_ids.append(fig_id)
        
        # Resolve the URL (handle relative paths)
        resolved_url = resolve_figure_url(
            fig_ref['url'],
            base_path=base_path,
            base_url=base_url
        )
        
        # Determine output filename
        ext = get_file_extension(fig_ref['url'])
        fig_filename = f"{fig_id}{ext}"
        fig_output_path = output_path / fig_filename
        
        figure_entry: Dict[str, Any] = {
            'id': fig_id,
            'description': fig_ref['alt'] or default_description,
            'image_path': str(fig_output_path),
            'source_url': fig_ref['url'],
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
                    base_path=base_path
                )
                logger.info("Downloaded %s: %s", fig_id, fig_ref['url'])
            except FigureDownloadError as e:
                download_errors.append(f"{fig_id}: {e}")
                logger.warning("Failed to download %s: %s", fig_id, e)
                figure_entry['download_error'] = str(e)
        
        figures.append(figure_entry)
    
    return figures, download_errors


def _report_loading_results(
    paper_title: str,
    paper_id: str,
    markdown_text: str,
    supplementary_text: Optional[str],
    figures: List[Dict[str, Any]],
    supplementary_figures: List[Dict[str, Any]],
    download_errors: List[str],
    download_figures: bool,
    length_warnings: List[str]
) -> None:
    """
    Log the results of paper loading.
    
    Args:
        paper_title: Extracted paper title
        paper_id: Paper identifier
        markdown_text: Main paper text
        supplementary_text: Supplementary text (may be None)
        figures: List of main figures
        supplementary_figures: List of supplementary figures
        download_errors: List of download error strings
        download_figures: Whether figures were downloaded
        length_warnings: Paper length warnings
    """
    total_text_length = len(markdown_text)
    total_tokens = estimate_tokens(markdown_text)
    if supplementary_text:
        total_text_length += len(supplementary_text)
        total_tokens += estimate_tokens(supplementary_text)
    
    logger.info("=" * 60)
    logger.info("Paper loaded from markdown:")
    logger.info("  Title: %s", paper_title)
    logger.info("  ID: %s", paper_id)
    logger.info("  Main text: %s chars (~%s tokens)", f"{len(markdown_text):,}", f"{estimate_tokens(markdown_text):,}")
    if supplementary_text:
        logger.info(
            "  Supplementary text: %s chars (~%s tokens)",
            f"{len(supplementary_text):,}",
            f"{estimate_tokens(supplementary_text):,}"
        )
    logger.info("  Total: %s chars (~%s tokens)", f"{total_text_length:,}", f"{total_tokens:,}")
    logger.info("  Main figures: %d", len(figures))
    if supplementary_figures:
        logger.info("  Supplementary figures: %d", len(supplementary_figures))
    if download_figures:
        total_figs = len(figures) + len(supplementary_figures)
        successful = total_figs - len(download_errors)
        logger.info("  Figures downloaded: %d/%d", successful, total_figs)
    logger.info("=" * 60)
    
    # Log length warnings
    for w in length_warnings:
        logger.warning("Length warning: %s", w)
    
    # Log download errors
    if download_errors:
        logger.warning("%d figure(s) failed to download:", len(download_errors))
        for err in download_errors:
            logger.warning("  - %s", err)


# ═══════════════════════════════════════════════════════════════════════
# Public Loading Functions
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
        logger.info("Paper input loaded with %d warning(s):", len(warnings))
        for w in warnings:
            logger.warning("  %s", w)
    
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
    # Build paper_input dict - validation will check types
    # Copy input lists and their contents to avoid side effects
    # Note: We pass figures as-is to let validation catch type errors
    # If it's a valid list, we'll copy it; otherwise validation will fail
    if isinstance(figures, list):
        figures_for_dict = [fig.copy() if isinstance(fig, dict) else fig for fig in figures]
    else:
        # Not a list - pass as-is, validation will catch this
        figures_for_dict = figures
    
    paper_input: Dict[str, Any] = {
        "paper_id": paper_id,
        "paper_title": paper_title,
        "paper_text": paper_text,
        "paper_domain": paper_domain,
        "figures": figures_for_dict,
    }
    
    # Build supplementary section if any supplementary content provided
    if supplementary_text or supplementary_figures or supplementary_data_files:
        supplementary: Dict[str, Any] = {}
        
        if supplementary_text:
            supplementary["supplementary_text"] = supplementary_text
        if supplementary_figures:
            # Copy supplementary figures list and contents to avoid side effects
            supplementary["supplementary_figures"] = [
                fig.copy() if isinstance(fig, dict) else fig for fig in supplementary_figures
            ]
        if supplementary_data_files:
            # Copy supplementary data files list and contents to avoid side effects
            supplementary["supplementary_data_files"] = [
                file.copy() if isinstance(file, dict) else file for file in supplementary_data_files
            ]
        
        paper_input["supplementary"] = supplementary
    
    validate_paper_input(paper_input)
    
    return paper_input


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
    # Check for empty path first
    if not markdown_path or not markdown_path.strip():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")
    
    md_path = Path(markdown_path)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")
    
    # Check if path is a directory, not a file
    if md_path.is_dir():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path} (path is a directory)")
    
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
    
    # Set up output directory with paper-specific and run-specific subfolders
    # Structure: {output_dir}/{paper_id}/run_{timestamp}/figures/
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"run_{run_timestamp}"
    paper_output_dir = Path(output_dir) / paper_id
    run_output_dir = paper_output_dir / run_folder_name
    figures_output_path = run_output_dir / "figures"
    figures_output_path.mkdir(parents=True, exist_ok=True)
    
    # Base path for resolving relative local paths
    md_base_path = md_path.parent
    
    # Process main paper figures
    logger.info("Processing %d main figure(s)...", len(figure_refs))
    figures, download_errors = _process_figure_refs(
        figure_refs=figure_refs,
        output_path=figures_output_path,
        base_path=md_base_path,
        base_url=base_url,
        download_figures=download_figures,
        figure_timeout=figure_timeout,
        existing_ids=[],
        id_prefix="",
        default_description="Figure from paper"
    )
    
    # Process supplementary materials if provided
    supplementary_text: Optional[str] = None
    supplementary_figures: List[Dict[str, Any]] = []
    
    if supplementary_markdown_path:
        supp_path = Path(supplementary_markdown_path)
        if not supp_path.exists():
            logger.warning("Supplementary file not found: %s", supplementary_markdown_path)
        else:
            # Read supplementary markdown
            with open(supp_path, 'r', encoding='utf-8') as f:
                supplementary_text = f.read()
            
            # Check supplementary length
            length_warnings.extend(check_paper_length(supplementary_text, "Supplementary"))
            
            # Extract supplementary figures
            supp_figure_refs = extract_figures_from_markdown(supplementary_text)
            supp_base_path = supp_path.parent
            supp_base = supplementary_base_url or base_url
            
            logger.info("Processing %d supplementary figure(s)...", len(supp_figure_refs))
            existing_main_ids = [f['id'] for f in figures]
            supplementary_figures, supp_errors = _process_figure_refs(
                figure_refs=supp_figure_refs,
                output_path=figures_output_path,
                base_path=supp_base_path,
                base_url=supp_base,
                download_figures=download_figures,
                figure_timeout=figure_timeout,
                existing_ids=existing_main_ids,
                id_prefix="S",
                default_description="Supplementary figure"
            )
            download_errors.extend(supp_errors)
    
    # Build PaperInput
    paper_input: Dict[str, Any] = {
        'paper_id': paper_id,
        'paper_title': paper_title,
        'paper_text': markdown_text,
        'paper_domain': paper_domain,
        'figures': figures,
        'run_output_dir': str(run_output_dir),
    }
    
    # Add supplementary section if we have any supplementary content
    if supplementary_text is not None or supplementary_figures:
        paper_input['supplementary'] = {}
        if supplementary_text is not None:
            paper_input['supplementary']['supplementary_text'] = supplementary_text
        if supplementary_figures:
            paper_input['supplementary']['supplementary_figures'] = supplementary_figures
    
    # Report results
    _report_loading_results(
        paper_title=paper_title,
        paper_id=paper_id,
        markdown_text=markdown_text,
        supplementary_text=supplementary_text,
        figures=figures,
        supplementary_figures=supplementary_figures,
        download_errors=download_errors,
        download_figures=download_figures,
        length_warnings=length_warnings
    )
    
    # Validate (will raise if critical errors)
    warnings = validate_paper_input(paper_input)
    if warnings:
        logger.info("Validation warnings:")
        for w in warnings:
            logger.warning("  %s", w)
    
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
    
    logger.info("Paper input saved to: %s", output_path)


# ═══════════════════════════════════════════════════════════════════════
# Accessor Functions
# ═══════════════════════════════════════════════════════════════════════

def get_figure_by_id(paper_input: PaperInput, figure_id: str) -> Optional[FigureInput]:
    """
    Get a specific figure from paper input by its ID.
    
    Args:
        paper_input: The paper input dictionary
        figure_id: The figure ID to find (e.g., "Fig3a")
        
    Returns:
        FigureInput dict if found, None otherwise
    """
    # Return None immediately if figure_id is None or empty string
    if not figure_id:
        return None
    
    for fig in paper_input.get("figures", []):
        # Only match if figure has an id field and it matches
        if "id" in fig and fig["id"] == figure_id:
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



