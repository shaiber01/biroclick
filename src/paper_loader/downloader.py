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
        if not url:
            raise FigureDownloadError("URL cannot be empty")
            
        parsed = urlparse(url)
        
        if parsed.scheme and parsed.scheme not in ('http', 'https', 'file'):
            raise FigureDownloadError(f"Unsupported scheme: {parsed.scheme}")
        
        if parsed.scheme in ('http', 'https'):
            # Remote URL - download it
            try:
                request = urllib.request.Request(
                    url,
                    headers={'User-Agent': DEFAULT_DOWNLOAD_CONFIG.user_agent}
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    content = response.read()
            except OSError as e:
                # Network/socket errors during download should be treated as download failures
                # This catches socket errors (which are OSError in Python 3) that might not be wrapped as URLError
                raise FigureDownloadError(f"Failed to download {url}: {e}")
            
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(content)
            except OSError as e:
                # File I/O errors during save
                raise FigureDownloadError(f"Failed to save figure to {output_path}: {e}")
                
        elif parsed.scheme == 'file':
            # file:// URL - extract path and copy
            # unquote handles %20 and other encoded chars
            local_path = Path(unquote(parsed.path)).expanduser()
            
            # On Windows, file:///C:/path becomes /C:/path which Path handles, 
            # but standard file:// URL handling can be tricky cross-platform.
            # Path(parsed.path) is usually robust enough.
            
            if not local_path.exists():
                raise FigureDownloadError(f"Local file not found: {local_path}")
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, output_path)
            
        else:
            # Assume local file path (relative or absolute)
            local_path = Path(url).expanduser()
            
            # If base_path provided, we enforce that the file must be within it
            # This prevents path traversal (../secret) and absolute path leaks (/etc/passwd)
            # when we expect a self-contained directory.
            if base_path:
                # Resolve both to get absolute paths with symlinks/.. removed
                # If local_path is absolute, /base / local_path is just local_path
                try:
                    if local_path.is_absolute():
                        resolved_path = local_path.resolve()
                    else:
                        resolved_path = (base_path / local_path).resolve()
                        
                    resolved_base = base_path.resolve()
                    
                    # Check if resolved path starts with resolved base path
                    if not resolved_path.is_relative_to(resolved_base):
                        raise FigureDownloadError(f"Access denied: Path {local_path} resolves to {resolved_path} which is outside base path {base_path}")
                        
                    local_path = resolved_path
                except (ValueError, RuntimeError) as e:
                     # is_relative_to raises ValueError if not relative
                     raise FigureDownloadError(f"Access denied: Path resolution failed: {e}")
            
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



