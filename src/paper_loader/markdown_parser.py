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
    
    Ignores images inside code blocks (```...```).
    
    Args:
        markdown_text: The markdown content
        
    Returns:
        List of dicts with 'alt', 'url', and 'original_match' keys
        
    Raises:
        TypeError: If markdown_text is not a string
    """
    if not isinstance(markdown_text, str):
        raise TypeError(f"markdown_text must be a string, got {type(markdown_text).__name__}")
    
    figures: List[Dict[str, str]] = []
    
    # First, identify code blocks to exclude them
    lines = markdown_text.split('\n')
    in_code_block = False
    code_block_ranges: List[tuple[int, int]] = []
    code_start = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('```'):
            if in_code_block:
                # End of code block
                code_block_ranges.append((code_start, i))
                in_code_block = False
            else:
                # Start of code block
                code_start = i
                in_code_block = True
    
    # If code block wasn't closed, include until end
    if in_code_block:
        code_block_ranges.append((code_start, len(lines)))
    
    # Build character ranges for code block content (more precise than line-based)
    # This ensures content after closing ``` on the same line is not considered in code block
    code_block_char_ranges: List[tuple[int, int]] = []
    
    # Calculate line start positions
    line_starts = [0]
    for line in lines[:-1]:  # All lines except last
        line_starts.append(line_starts[-1] + len(line) + 1)  # +1 for newline
    
    for start_line, end_line in code_block_ranges:
        # Start of code block is at the beginning of the start line
        block_start = line_starts[start_line]
        
        # End of code block: find the position right after the closing ```
        if end_line < len(lines):
            closing_line = lines[end_line]
            fence_pos = closing_line.find('```')
            if fence_pos != -1:
                # Include up to and including the closing ```
                block_end = line_starts[end_line] + fence_pos + 3
            else:
                # No closing fence found (unclosed block), include to end of line
                block_end = line_starts[end_line] + len(closing_line)
        else:
            # end_line is beyond lines (unclosed block at end)
            block_end = len(markdown_text)
        
        code_block_char_ranges.append((block_start, block_end))
    
    # Helper function to check if a position is in a code block
    def is_in_code_block(char_pos: int) -> bool:
        for start_char, end_char in code_block_char_ranges:
            if start_char <= char_pos < end_char:
                return True
        return False
    
    # Pattern for markdown images: ![alt](url) or ![alt](url "title")
    # Supports one level of nested brackets in alt text and parentheses/spaces in URL
    # URL can contain spaces and nested parentheses, but we stop at closing paren or quoted title
    # Match URL: allow nested parentheses and spaces, stop before optional title
    # Note: Alt text and URL should not span multiple lines (standard markdown behavior)
    markdown_pattern = r'!\[((?:[^\[\]\n]|\[[^\]\n]*\])*)\]\(((?:[^()\n]|\([^)\n]*\))+?)(?:\s+"[^"\n]*")?\)'
    for match in re.finditer(markdown_pattern, markdown_text):
        # Skip if match is inside a code block
        if is_in_code_block(match.start()):
            continue
            
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
        # Skip if match is inside a code block
        if is_in_code_block(match.start()):
            continue
            
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
    1. If URL is absolute (http/https/file/data URI), use as-is
    2. If base_url is provided and URL is relative, join with base_url
    3. If base_path is provided and URL is relative, resolve against base_path
    4. Otherwise, return URL as-is
    
    Args:
        url: The figure URL or path from markdown
        base_path: Optional base path (typically the markdown file's directory)
        base_url: Optional base URL for remote relative paths
        
    Returns:
        Resolved URL or path string
        
    Raises:
        TypeError: If url is not a string
    """
    # Validate input type
    if not isinstance(url, str):
        raise TypeError(f"url must be a string, got {type(url).__name__}")
    
    parsed = urlparse(url)
    
    # Already absolute URL (including data URIs which contain inline image data)
    if parsed.scheme in ('http', 'https', 'file', 'data'):
        return url
    
    # Relative URL with base_url provided
    if base_url and not parsed.scheme:
        # Normalize base_url to end with / to ensure proper directory joining
        # urljoin treats the last component as a file if base_url doesn't end with /
        # For relative URLs (not starting with /), we want directory-like behavior
        # unless base_url clearly ends with a file (has a file extension)
        normalized_base_url = base_url
        if not url.startswith('/') and not normalized_base_url.endswith('/'):
            # Check if base_url ends with a file extension
            parsed_base = urlparse(base_url)
            base_path = parsed_base.path
            if base_path:
                last_component = base_path.rstrip('/').split('/')[-1]
                # Common file extensions - if present, don't add trailing slash
                # (let urljoin replace the file)
                has_extension = '.' in last_component and last_component.split('.')[-1].lower() in (
                    'html', 'htm', 'php', 'asp', 'aspx', 'jsp', 'png', 'jpg', 'jpeg', 
                    'gif', 'svg', 'pdf', 'txt', 'xml', 'json', 'css', 'js'
                )
                if not has_extension:
                    # No file extension detected, treat as directory
                    normalized_base_url = base_url + '/'
        return urljoin(normalized_base_url, url)
    
    # Relative path with base_path provided
    if base_path and not parsed.scheme:
        # Decode URL-encoded characters for file paths
        decoded_url = unquote(url)
        resolved = (base_path / decoded_url).resolve()
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
    # Improved regex to handle space/period separators better
    # Matches: Fig 1, Fig. 1, Figure 1, Fig S1, Figure 3(a), Fig. 2-3
    fig_match = re.search(r'(?:fig(?:ure)?|fig\.?)\s*([a-z]?\d+(?:[.\-]\d+)?[a-z]?(?:\([a-z]\))?)', alt_text, re.IGNORECASE)
    if fig_match:
        return f"Fig{fig_match.group(1)}"
    
    # Also try "Fig. S1" pattern where S is separate
    # Matches: Fig. S1, Figure S3
    fig_s_match = re.search(r'(?:fig(?:ure)?|fig\.?)\s+(s\d+[a-z]?)', alt_text, re.IGNORECASE)
    if fig_s_match:
        return f"Fig{fig_s_match.group(1)}"

    # Try to extract from URL filename
    parsed = urlparse(url)
    filename = Path(unquote(parsed.path)).stem
    # Allow spaces and other separators in filename (e.g., "figure 13" from "figure%2013")
    fig_match = re.search(r'fig(?:ure)?[_\- ]*(\d+(?:[.\-]\d+)?[a-z]?)', filename, re.IGNORECASE)
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
    1. First H1 heading: # Title (ignoring code blocks)
    2. HTML h1 tag: <h1>Title</h1>
    3. First non-empty, non-image, non-codeblock line
    
    Args:
        markdown_text: The markdown content
        
    Returns:
        Paper title string, or "Untitled Paper" if not found
    """
    lines = markdown_text.split('\n')
    in_code_block = False
    
    # Pass 1: Look for H1 heading outside code blocks
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
            
        if in_code_block:
            # Skip everything inside code blocks
            # We'll handle unclosed code blocks by checking after the loop
            continue
            
        # Check for H1 # Title (hash followed by whitespace)
        # Handle both cases: # Title (with content) and # or #  (empty/no content)
        if stripped == '#':
            # H1 with no content
            return ''
        h1_match = re.match(r'^#\s+(.*)$', stripped)
        if h1_match:
            title = h1_match.group(1).strip()
            # Return title (empty string if H1 had only whitespace)
            return title
    
    # If we're still in a code block, it's unclosed
    # Re-process looking for H1 headings, but skip lines that are clearly code content
    if in_code_block:
        # Find where the code block started
        code_block_start_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('```'):
                code_block_start_idx = i
                break
        
        # Look for H1 headings after the code block start
        # Skip H1s that are immediately after code block start (likely code comments)
        # Process H1 headings that come after blank lines or other content (likely actual titles)
        if code_block_start_idx >= 0:
            seen_blank_after_code = False
            for i in range(code_block_start_idx + 1, len(lines)):
                stripped = lines[i].strip()
                # Skip code block markers
                if stripped.startswith('```'):
                    continue
                # Track if we've seen blank lines (indicates we've left code content area)
                if not stripped:
                    seen_blank_after_code = True
                    continue
                # Look for H1 headings
                # Only process H1s that come after blank lines (not code comments)
                if seen_blank_after_code:
                    h1_match = re.match(r'^#\s+(.+)$', stripped)
                    if h1_match:
                        return h1_match.group(1).strip()
            
    # Pass 2: Look for HTML h1 (regex might still be fooled by code blocks but unlikely to match exactly)
    html_h1_match = re.search(r'<h1[^>]*>(.+?)</h1>', markdown_text, re.IGNORECASE | re.DOTALL)
    if html_h1_match:
        # Strip any HTML tags inside
        title = re.sub(r'<[^>]+>', '', html_h1_match.group(1))
        return title.strip()
    
    # Pass 3: Look for first non-empty line as fallback
    # Re-detect code blocks for this pass to handle unclosed blocks correctly
    in_code_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
            
        if in_code_block:
            continue
            
        line = line.strip()
        if line and not line.startswith('!') and not line.startswith('<'):
            return line[:200]  # Truncate very long lines
    
    return "Untitled Paper"



