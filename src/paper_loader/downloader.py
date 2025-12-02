"""
Figure download functionality.

This module handles downloading figures from various sources:
- Remote URLs (http, https)
- Local file paths (absolute or relative)
- file:// URLs
"""

import shutil
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote

from .config import DEFAULT_DOWNLOAD_CONFIG


class FigureDownloadError(Exception):
    """Raised when a figure cannot be downloaded."""
    pass


def download_figure(
    url: str,
    output_path: Path,
    timeout: Optional[int] = None,
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
        timeout: Download timeout in seconds (for remote URLs).
                 Defaults to DEFAULT_DOWNLOAD_CONFIG.timeout_seconds
        base_path: Base path for resolving relative local paths
        
    Raises:
        FigureDownloadError: If download/copy fails
    """
    if timeout is None:
        timeout = DEFAULT_DOWNLOAD_CONFIG.timeout_seconds
    
    try:
        parsed = urlparse(url)
        
        if parsed.scheme in ('http', 'https'):
            # Remote URL - download it
            request = urllib.request.Request(
                url,
                headers={'User-Agent': DEFAULT_DOWNLOAD_CONFIG.user_agent}
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



