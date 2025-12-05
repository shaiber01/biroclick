"""
Configuration constants for the paper_loader package.

This module centralizes all constants, thresholds, and configuration
used throughout the paper loading system.
"""

from dataclasses import dataclass
from typing import Dict, Set


# ═══════════════════════════════════════════════════════════════════════
# Supported Image Formats
# ═══════════════════════════════════════════════════════════════════════

# Image formats supported by vision-capable LLMs (Claude, GPT-4, etc.)
SUPPORTED_IMAGE_FORMATS: Dict[str, str] = {
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
VISION_MODEL_PREFERRED_FORMATS: Set[str] = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}


# ═══════════════════════════════════════════════════════════════════════
# Paper Length Thresholds
# ═══════════════════════════════════════════════════════════════════════

# Character thresholds for paper length warnings
PAPER_LENGTH_NORMAL = 50_000       # ~12,500 tokens - typical paper
PAPER_LENGTH_LONG = 150_000        # ~37,500 tokens - long paper, consider trimming
PAPER_LENGTH_VERY_LONG = 300_000   # ~75,000 tokens - very long, may hit context limits

# Characters per token estimate (for English text)
CHARS_PER_TOKEN = 4


# ═══════════════════════════════════════════════════════════════════════
# Token Cost Estimation Constants
# ═══════════════════════════════════════════════════════════════════════

# ~170 tokens per 512x512 tile, typical figure = ~4 tiles = 680 tokens
TOKENS_PER_FIGURE = 680

# Cost per million tokens (approximate - verify current rates)
INPUT_COST_PER_MILLION = 3.00   # Conservative estimate
OUTPUT_COST_PER_MILLION = 10.00


# ═══════════════════════════════════════════════════════════════════════
# Valid Domains
# ═══════════════════════════════════════════════════════════════════════

VALID_DOMAINS = [
    "plasmonics",
    "photonic_crystal",
    "metamaterial",
    "thin_film",
    "waveguide",
    "strong_coupling",
    "nonlinear",
    "other"
]


# ═══════════════════════════════════════════════════════════════════════
# Image Validation Thresholds
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ImageValidationConfig:
    """Configuration for image validation thresholds."""
    max_file_size_mb: float = 5.0
    min_resolution: int = 512
    max_resolution: int = 4096
    max_aspect_ratio: float = 5.0


# Default image validation configuration
DEFAULT_IMAGE_CONFIG = ImageValidationConfig()


# ═══════════════════════════════════════════════════════════════════════
# Download Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DownloadConfig:
    """Configuration for figure downloads."""
    timeout_seconds: int = 30
    user_agent: str = 'Mozilla/5.0 (compatible; PaperLoader/1.0)'


# Default download configuration
DEFAULT_DOWNLOAD_CONFIG = DownloadConfig()






