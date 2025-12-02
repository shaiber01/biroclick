"""
Paper Input Loader for Paper Reproduction System

This package handles loading and validating paper inputs including
text, figures, and optional digitized data.

Supports two loading modes:
1. JSON-based: Load from a pre-prepared JSON file with explicit figure paths
2. Markdown-based: Parse markdown (from marker/nougat), extract and download figures

All public symbols are re-exported here for backward compatibility:
    from src.paper_loader import load_paper_from_markdown, PaperInput, ValidationError
"""

# Config - constants and configuration
from .config import (
    SUPPORTED_IMAGE_FORMATS,
    VISION_MODEL_PREFERRED_FORMATS,
    PAPER_LENGTH_NORMAL,
    PAPER_LENGTH_LONG,
    PAPER_LENGTH_VERY_LONG,
    CHARS_PER_TOKEN,
    TOKENS_PER_FIGURE,
    INPUT_COST_PER_MILLION,
    OUTPUT_COST_PER_MILLION,
    VALID_DOMAINS,
    ImageValidationConfig,
    DownloadConfig,
    DEFAULT_IMAGE_CONFIG,
    DEFAULT_DOWNLOAD_CONFIG,
)

# Schemas - type definitions
from .schemas import (
    FigureInput,
    DataFileInput,
    SupplementaryInput,
    PaperInput,
    EXAMPLE_PAPER_INPUT,
)

# Validation
from .validation import (
    ValidationError,
    validate_paper_input,
    validate_domain,
    validate_figure_image,
)

# Cost estimation
from .cost_estimation import (
    estimate_tokens,
    estimate_token_cost,
    check_paper_length,
)

# Markdown parsing
from .markdown_parser import (
    extract_figures_from_markdown,
    extract_paper_title,
    resolve_figure_url,
    generate_figure_id,
    get_file_extension,
)

# Downloader
from .downloader import (
    FigureDownloadError,
    download_figure,
)

# State conversion
from .state_conversion import (
    create_state_from_paper_input,
    load_paper_text,
)

# Loaders and accessors
from .loaders import (
    load_paper_input,
    load_paper_from_markdown,
    create_paper_input,
    save_paper_input_json,
    get_figure_by_id,
    list_figure_ids,
    get_supplementary_text,
    get_supplementary_figures,
    get_supplementary_data_files,
    get_data_file_by_type,
    get_all_figures,
)


__all__ = [
    # Config
    "SUPPORTED_IMAGE_FORMATS",
    "VISION_MODEL_PREFERRED_FORMATS",
    "PAPER_LENGTH_NORMAL",
    "PAPER_LENGTH_LONG",
    "PAPER_LENGTH_VERY_LONG",
    "CHARS_PER_TOKEN",
    "TOKENS_PER_FIGURE",
    "INPUT_COST_PER_MILLION",
    "OUTPUT_COST_PER_MILLION",
    "VALID_DOMAINS",
    "ImageValidationConfig",
    "DownloadConfig",
    "DEFAULT_IMAGE_CONFIG",
    "DEFAULT_DOWNLOAD_CONFIG",
    # Schemas
    "FigureInput",
    "DataFileInput",
    "SupplementaryInput",
    "PaperInput",
    "EXAMPLE_PAPER_INPUT",
    # Validation
    "ValidationError",
    "validate_paper_input",
    "validate_domain",
    "validate_figure_image",
    # Cost estimation
    "estimate_tokens",
    "estimate_token_cost",
    "check_paper_length",
    # Markdown parsing
    "extract_figures_from_markdown",
    "extract_paper_title",
    "resolve_figure_url",
    "generate_figure_id",
    "get_file_extension",
    # Downloader
    "FigureDownloadError",
    "download_figure",
    # State conversion
    "create_state_from_paper_input",
    "load_paper_text",
    # Loaders and accessors
    "load_paper_input",
    "load_paper_from_markdown",
    "create_paper_input",
    "save_paper_input_json",
    "get_figure_by_id",
    "list_figure_ids",
    "get_supplementary_text",
    "get_supplementary_figures",
    "get_supplementary_data_files",
    "get_data_file_by_type",
    "get_all_figures",
]
