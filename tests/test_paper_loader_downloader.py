"""
Unit tests for paper_loader downloader module.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import io

from src.paper_loader import (
    FigureDownloadError,
    download_figure,
)


# ═══════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "paper_loader"


# ═══════════════════════════════════════════════════════════════════════
# download_figure Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDownloadFigure:
    """Tests for download_figure function."""
    
    def test_local_file_copy(self, tmp_path):
        """Downloads (copies) local file to output path."""
        source = FIXTURES_DIR / "sample_images" / "test_figure.png"
        output = tmp_path / "output" / "copied.png"
        
        download_figure(str(source), output, base_path=FIXTURES_DIR)
        
        assert output.exists()
        assert output.stat().st_size == source.stat().st_size
    
    def test_local_file_relative_with_base_path(self, tmp_path):
        """Resolves relative path against base_path."""
        output = tmp_path / "fig.png"
        
        download_figure(
            "sample_images/test_figure.png",
            output,
            base_path=FIXTURES_DIR
        )
        
        assert output.exists()
    
    def test_local_file_creates_parent_dirs(self, tmp_path):
        """Creates parent directories if they don't exist."""
        source = FIXTURES_DIR / "sample_images" / "test_figure.png"
        output = tmp_path / "deep" / "nested" / "path" / "fig.png"
        
        download_figure(str(source), output)
        
        assert output.exists()
    
    def test_local_file_not_found_raises(self, tmp_path):
        """Raises FigureDownloadError for non-existent local file."""
        output = tmp_path / "output.png"
        
        with pytest.raises(FigureDownloadError, match="Local file not found"):
            download_figure("/nonexistent/file.png", output)
    
    def test_file_url_copy(self, tmp_path):
        """Handles file:// URL scheme."""
        source = FIXTURES_DIR / "sample_images" / "test_figure.png"
        output = tmp_path / "from_file_url.png"
        
        download_figure(f"file://{source}", output)
        
        assert output.exists()
    
    def test_file_url_not_found_raises(self, tmp_path):
        """Raises FigureDownloadError for non-existent file:// URL."""
        output = tmp_path / "output.png"
        
        with pytest.raises(FigureDownloadError, match="Local file not found"):
            download_figure("file:///nonexistent/path/image.png", output)
    
    @patch('urllib.request.urlopen')
    def test_http_download(self, mock_urlopen, tmp_path):
        """Downloads figure from HTTP URL."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.read.return_value = b'\x89PNG\r\n\x1a\n'  # PNG magic bytes
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        output = tmp_path / "downloaded.png"
        
        download_figure("http://example.com/image.png", output)
        
        assert output.exists()
        mock_urlopen.assert_called_once()
    
    @patch('urllib.request.urlopen')
    def test_https_download(self, mock_urlopen, tmp_path):
        """Downloads figure from HTTPS URL."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'image data'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        output = tmp_path / "secure.png"
        
        download_figure("https://example.com/secure.png", output)
        
        assert output.exists()
    
    @patch('urllib.request.urlopen')
    def test_http_url_error_raises(self, mock_urlopen, tmp_path):
        """Raises FigureDownloadError on URL error."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection failed")
        
        output = tmp_path / "failed.png"
        
        with pytest.raises(FigureDownloadError, match="Failed to download"):
            download_figure("http://example.com/image.png", output)
    
    @patch('urllib.request.urlopen')
    def test_http_error_raises(self, mock_urlopen, tmp_path):
        """Raises FigureDownloadError on HTTP error."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "http://example.com", 404, "Not Found", {}, None
        )
        
        output = tmp_path / "notfound.png"
        
        with pytest.raises(FigureDownloadError, match="HTTP Error"):
            download_figure("http://example.com/missing.png", output)
    
    @patch('urllib.request.urlopen')
    def test_sets_user_agent(self, mock_urlopen, tmp_path):
        """Sets User-Agent header for HTTP requests."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'data'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        output = tmp_path / "test.png"
        download_figure("https://example.com/fig.png", output)
        
        # Check that urlopen was called with a Request object
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert hasattr(request, 'get_header')
        assert 'User-agent' in request.headers or 'user-agent' in [k.lower() for k in request.headers]
    
    def test_default_timeout_used(self, tmp_path):
        """Default timeout is applied (tested via signature)."""
        # This test verifies the function accepts timeout parameter
        source = FIXTURES_DIR / "sample_images" / "test_figure.png"
        output = tmp_path / "fig.png"
        
        # Should not raise
        download_figure(str(source), output, timeout=60)
        
        assert output.exists()


class TestFigureDownloadError:
    """Tests for FigureDownloadError exception."""
    
    def test_is_exception(self):
        """FigureDownloadError is an Exception."""
        assert issubclass(FigureDownloadError, Exception)
    
    def test_has_message(self):
        """FigureDownloadError stores message."""
        error = FigureDownloadError("Download failed")
        assert str(error) == "Download failed"

