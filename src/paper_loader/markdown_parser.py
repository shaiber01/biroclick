"""
Markdown parsing utilities for extracting paper content and figures.

This module provides functions to:
- Extract figure references from markdown (![alt](url) and <img> tags)
- Extract paper titles from markdown headings
- Resolve relative URLs against base paths
- Generate figure IDs from alt text or filenames
- Determine file extensions from URLs
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse, unquote, urljoin

from .config import SUPPORTED_IMAGE_FORMATS


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
    figures: List[Dict[str, str]] = []
    
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


def resolve_figure_url(
    url: str,
    base_path: Optional[Path] = None,
    base_url: Optional[str] = None
) -> str:
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


