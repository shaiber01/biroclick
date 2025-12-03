"""
Validation functions for paper input data.

This module provides validation for paper inputs, figures, and domains.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from schemas.state import CONTEXT_WINDOW_LIMITS

from .config import (
    VALID_DOMAINS,
    VISION_MODEL_PREFERRED_FORMATS,
    DEFAULT_IMAGE_CONFIG,
)


logger = logging.getLogger(__name__)


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
    errors: List[str] = []
    warnings: List[str] = []
    
    # Check required fields
    required_fields = ["paper_id", "paper_title", "paper_text", "figures"]
    for field in required_fields:
        if field not in paper_input:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        # If critical required fields are missing, we might want to stop here
        # because further validation might crash (e.g. None types)
        # But let's see if we can continue safely.
        # The original code raised here. To accumulate more errors, we need to be careful.
        # For now, we will continue but check for existence before validating.
        pass
    
    # Validate paper_id format
    if "paper_id" in paper_input:
        paper_id = paper_input["paper_id"]
        if not paper_id or not isinstance(paper_id, str):
            errors.append("paper_id must be a non-empty string")
        elif " " in paper_id:
            warnings.append(f"paper_id contains spaces: '{paper_id}' - consider using underscores")
    
    # Validate paper_title format
    if "paper_title" in paper_input:
        paper_title = paper_input["paper_title"]
        if paper_title is None or not isinstance(paper_title, str):
            errors.append("paper_title must be a string")
    
    # Validate paper_text is not empty
    if "paper_text" in paper_input:
        paper_text = paper_input["paper_text"]
        if not paper_text or not isinstance(paper_text, str) or len(paper_text.strip()) < 100:
            errors.append("paper_text is empty or too short (< 100 chars)")
        else:
            # Validate paper_text is not too long (v1: hard limit, no auto-trimming)
            # See CONTEXT_WINDOW_LIMITS in schemas/state.py for the canonical source
            max_chars = CONTEXT_WINDOW_LIMITS["max_paper_chars"]
            paper_length = len(paper_text)
            if paper_length > max_chars:
                errors.append(
                    f"Paper exceeds maximum length ({max_chars:,} chars). "
                    f"Current length: {paper_length:,} chars (~{paper_length // 4:,} tokens).\n\n"
                    f"Please manually trim the paper before loading:\n"
                    f"1. Remove the References section\n"
                    f"2. Remove Acknowledgments, Author Contributions, Funding sections\n"
                    f"3. Remove detailed literature review paragraphs\n\n"
                    f"DO NOT remove: Methods, Results, Figure captions, or Key equations.\n\n"
                    f"Future versions will support automatic trimming and chunking."
                )
    
    # Validate figures
    if "figures" in paper_input:
        figures = paper_input["figures"]
        if not isinstance(figures, list):
            errors.append("figures must be a list")
        else:
            if len(figures) == 0:
                warnings.append("No figures provided - system needs figure images for visual comparison")
            
            for i, fig in enumerate(figures):
                # Check required figure fields
                if not isinstance(fig, dict):
                     errors.append(f"Figure {i}: must be a dictionary")
                     continue

                if "id" not in fig:
                    errors.append(f"Figure {i}: missing 'id' field")
                elif not isinstance(fig["id"], str):
                    errors.append(f"Figure {i}: 'id' must be a string, got {type(fig['id']).__name__}")
                elif len(fig["id"]) == 0:
                    errors.append(f"Figure {i}: 'id' must be non-empty")
                
                # image_path is optional if source_url is present, but usually required for processing
                # Let's check strict requirement:
                if "image_path" not in fig:
                    errors.append(f"Figure {i} ({fig.get('id', 'unknown')}): missing 'image_path' field")
                else:
                    # Check if image file exists
                    img_path = Path(fig["image_path"])
                    if not img_path.exists():
                        warnings.append(f"Figure {fig.get('id', i)}: image file not found: {fig['image_path']}")
                    elif img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                        warnings.append(f"Figure {fig.get('id', i)}: unusual image format: {img_path.suffix}")
                
                # Check digitized data if provided
                if fig.get("digitized_data_path"):
                    data_path = Path(fig["digitized_data_path"])
                    if not data_path.exists():
                        warnings.append(f"Figure {fig.get('id', i)}: digitized data file not found: {fig['digitized_data_path']}")
    
    # Validate supplementary figures (same validation as main figures)
    supplementary = paper_input.get("supplementary", {})
    if "supplementary_figures" in supplementary:
        supp_figures = supplementary["supplementary_figures"]
        if not isinstance(supp_figures, list):
            errors.append("supplementary_figures must be a list")
        else:
            for i, fig in enumerate(supp_figures):
                # Check required figure fields
                if not isinstance(fig, dict):
                    errors.append(f"Supplementary figure {i}: must be a dictionary")
                    continue

                if "id" not in fig:
                    errors.append(f"Supplementary figure {i}: missing 'id' field")
                elif not isinstance(fig["id"], str):
                    errors.append(f"Supplementary figure {i}: 'id' must be a string, got {type(fig['id']).__name__}")
                elif len(fig["id"]) == 0:
                    errors.append(f"Supplementary figure {i}: 'id' must be non-empty")
                
                if "image_path" not in fig:
                    errors.append(f"Supplementary figure {i} ({fig.get('id', 'unknown')}): missing 'image_path' field")
                else:
                    # Check if image file exists
                    img_path = Path(fig["image_path"])
                    if not img_path.exists():
                        warnings.append(f"Supplementary figure {fig.get('id', i)}: image file not found: {fig['image_path']}")
                    elif img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                        warnings.append(f"Supplementary figure {fig.get('id', i)}: unusual image format: {img_path.suffix}")
    
    # Validate supplementary data files
    if "supplementary_data_files" in supplementary:
        supp_data_files = supplementary["supplementary_data_files"]
        if not isinstance(supp_data_files, list):
            errors.append("supplementary_data_files must be a list")
        else:
            for i, data_file in enumerate(supp_data_files):
                if not isinstance(data_file, dict):
                    errors.append(f"Supplementary data file {i}: must be a dictionary")
                    continue
                
                required_fields = ["id", "description", "file_path", "data_type"]
                for field in required_fields:
                    if field not in data_file:
                        errors.append(f"Supplementary data file {i} ({data_file.get('id', 'unknown')}): missing '{field}' field")
    
    if errors:
        raise ValidationError(f"Paper input validation failed:\n" + "\n".join(errors))
    
    return warnings


def validate_domain(domain: str) -> bool:
    """
    Check if domain is one of the recognized values.
    
    Args:
        domain: Domain string to validate
        
    Returns:
        True if domain is valid, False otherwise
    """
    return domain in VALID_DOMAINS


def validate_figure_image(image_path: str) -> List[str]:
    """
    Check if a figure image is suitable for vision models.
    
    Vision-capable LLMs work best with certain image characteristics.
    This function checks for common issues that may affect comparison quality.
    
    Args:
        image_path: Path to the figure image file
        
    Returns:
        List of warnings (empty if image is suitable)
    """
    warnings: List[str] = []
    path = Path(image_path)
    config = DEFAULT_IMAGE_CONFIG
    
    if not path.exists():
        warnings.append(f"Image file not found: {image_path}")
        return warnings
    
    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > config.max_file_size_mb:
        warnings.append(f"Large file size ({size_mb:.1f}MB) - consider compressing to save tokens")
    
    # Try to check image dimensions using PIL if available
    try:
        from PIL import Image
        
        img = Image.open(path)
        width, height = img.size
        
        if width < config.min_resolution or height < config.min_resolution:
            warnings.append(
                f"Low resolution ({width}x{height}) - vision models work best with "
                f"≥{config.min_resolution}px on each side"
            )
        
        if width > config.max_resolution or height > config.max_resolution:
            warnings.append(
                f"Very high resolution ({width}x{height}) - consider resizing to "
                f"≤{config.max_resolution}px to save tokens"
            )
        
        # Check for very unusual aspect ratios
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > config.max_aspect_ratio:
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



