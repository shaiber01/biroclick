"""
Unit tests for paper_loader downloader module.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import urllib.error
import stat

from src.paper_loader.downloader import (
    FigureDownloadError,
    download_figure,
)
from src.paper_loader.config import DEFAULT_DOWNLOAD_CONFIG

# ═══════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "paper_loader"

@pytest.fixture
def mock_urlopen():
    with patch('urllib.request.urlopen') as mock:
        yield mock

# ═══════════════════════════════════════════════════════════════════════
# download_figure Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDownloadFigure:
    """Tests for download_figure function."""
    
    # --- Input Validation ---

    def test_empty_url_raises_error(self, tmp_path):
        """Raises FigureDownloadError when URL is empty."""
        output = tmp_path / "out.png"
        with pytest.raises(FigureDownloadError):
            download_figure("", output)

    def test_unsupported_scheme_raises_informative_error(self, tmp_path):
        """
        Raises FigureDownloadError for unsupported schemes (e.g. ftp).
        Should not treat ftp:// as a local file path.
        """
        output = tmp_path / "out.png"
        # The current implementation falls through to local file handling for unknown schemes.
        # This test asserts that we explicitly reject unknown schemes or handle them properly.
        # If the code treats 'ftp://...' as a local file, it will raise "Local file not found",
        # which is confusing. We want a clearer error or explicit validation.
        
        url = "ftp://example.com/image.png"
        with pytest.raises(FigureDownloadError) as excinfo:
            download_figure(url, output)
        
        # We want to ensure it didn't try to look for a local file named "ftp://..."
        # So the error message should NOT say "Local file not found" or should explicitly say "Unsupported scheme"
        msg = str(excinfo.value)
        assert "Local file not found" not in msg, f"Bug: Treated {url} as local file. Error: {msg}"
        assert "scheme" in msg.lower() or "supported" in msg.lower()

    # --- Local File Handling ---

    def test_local_file_absolute_path_ignores_base_path(self, tmp_path):
        """Absolute path should ignore base_path if provided."""
        # Create a dummy file in tmp_path
        source = tmp_path / "source.png"
        source.write_text("content")
        
        output = tmp_path / "output.png"
        
        # Provide a base_path that does NOT contain the file
        base_path = tmp_path / "other_dir"
        base_path.mkdir()
        
        # Should use absolute path 'source' and ignore 'base_path'
        download_figure(str(source.absolute()), output, base_path=base_path)
        
        assert output.exists()
        assert output.read_text() == "content"

    def test_local_file_tilde_expansion(self, tmp_path):
        """
        Should expand user tilde (~).
        Note: This might fail if the code doesn't use expanduser().
        """
        # We can't easily create a file in actual user home, but we can mock Path.expanduser
        # or check if the code calls it.
        # Instead, we'll pass a path starting with ~ and assert that it tries to expand it
        # rather than looking for a literal '~' directory.
        
        with patch('pathlib.Path.expanduser') as mock_expand:
            mock_expand.return_value = tmp_path / "expanded.png"
            
            # We need the file to exist at the expanded path
            (tmp_path / "expanded.png").write_text("data")
            
            output = tmp_path / "out.png"
            
            # If code doesn't call expanduser, it will look for literal "~/..." and fail
            # or if it does call it, it will use our mock and succeed.
            
            # NOTE: We only test that it succeeds if expansion is handled.
            # If implementation lacks expanduser(), this test will fail (file not found).
            download_figure("~/expanded.png", output)
            
            assert output.exists()

    def test_local_file_permissions_error(self, tmp_path):
        """Raises FigureDownloadError on read permission denied."""
        source = tmp_path / "locked.png"
        source.write_text("secret")
        
        # Remove read permissions
        source.chmod(0o000)
        
        output = tmp_path / "out.png"
        
        try:
            with pytest.raises(FigureDownloadError) as exc:
                download_figure(str(source), output)
            assert "Permission denied" in str(exc.value) or "eacces" in str(exc.value).lower()
        finally:
            source.chmod(0o666) # Restore for cleanup

    def test_output_directory_creation_failure(self, tmp_path):
        """Raises FigureDownloadError if output directory cannot be created."""
        # Create a file where the directory should be
        (tmp_path / "blocked").write_text("im a file")
        output = tmp_path / "blocked" / "file.png"
        source = tmp_path / "source.png"
        source.write_text("data")

        with pytest.raises(FigureDownloadError) as exc:
            download_figure(str(source), output)
        
        # Should catch OSError/FileExistsError and wrap it
        assert "Failed to save" in str(exc.value)

    # --- File URL Handling ---

    def test_file_url_decoding(self, tmp_path):
        """Decodes URL encoded characters in file:// paths."""
        # Create file with spaces/special chars
        dir_name = "folder with spaces"
        file_name = "image #1.png"
        (tmp_path / dir_name).mkdir()
        source = tmp_path / dir_name / file_name
        source.write_text("data")
        
        # Encode path: spaces -> %20, # -> %23
        # naive encoding for test
        url_path = str(source).replace(" ", "%20").replace("#", "%23")
        if not url_path.startswith("/"):
            url_path = "/" + url_path
        url = f"file://{url_path}"
        
        output = tmp_path / "out.png"
        download_figure(url, output)
        
        assert output.exists()
        assert output.read_text() == "data"

    # --- HTTP/HTTPS Handling ---

    def test_http_download_uses_correct_timeout(self, mock_urlopen, tmp_path):
        """Verifies the timeout parameter is passed to urlopen."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'data'
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        output = tmp_path / "out.png"
        
        # Custom timeout
        download_figure("http://example.com/img.png", output, timeout=42)
        
        args, kwargs = mock_urlopen.call_args
        assert kwargs.get('timeout') == 42

    def test_http_download_uses_default_timeout(self, mock_urlopen, tmp_path):
        """Verifies default timeout is used when none provided."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'data'
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        output = tmp_path / "out.png"
        
        download_figure("http://example.com/img.png", output)
        
        args, kwargs = mock_urlopen.call_args
        assert kwargs.get('timeout') == DEFAULT_DOWNLOAD_CONFIG.timeout_seconds

    def test_http_download_content_verification(self, mock_urlopen, tmp_path):
        """Verifies that the exact content read from response is written to file."""
        expected_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        
        mock_response = MagicMock()
        mock_response.read.return_value = expected_content
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        output = tmp_path / "img.png"
        download_figure("http://example.com/img.png", output)
        
        assert output.read_bytes() == expected_content

    def test_http_404_raises_specific_error(self, mock_urlopen, tmp_path):
        """Raises FigureDownloadError containing status code for 404."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 404, "Not Found", {}, None
        )
        
        output = tmp_path / "out.png"
        
        with pytest.raises(FigureDownloadError) as exc:
            download_figure("http://example.com/missing.png", output)
        
        assert "404" in str(exc.value)
        assert "Not Found" in str(exc.value)

    def test_http_bad_response_handling(self, mock_urlopen, tmp_path):
        """
        Test handling of incomplete reads or connection resets during read.
        This simulates an IncompleteRead exception during response.read().
        """
        mock_response = MagicMock()
        # Simulate read failing halfway
        mock_response.read.side_effect = urllib.error.URLError("Connection reset")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        output = tmp_path / "out.png"
        
        with pytest.raises(FigureDownloadError) as exc:
            download_figure("http://example.com/img.png", output)
        
        assert "Failed to download" in str(exc.value)
        # Ensure partial file is not left (optional but good practice) or just ensure it failed.

    # --- Security/Safety ---
    
    def test_path_traversal_prevention(self, tmp_path):
        """
        Should prevent writing outside output directory? 
        Actually, output_path is provided by caller, so caller controls destination.
        But source path traversal for local files might be a concern if source is untrusted.
        
        This test checks if we can read a file outside base_path using relative paths 
        when base_path is SET.
        """
        # Create a secret file outside base_path
        secret_dir = tmp_path / "secret"
        secret_dir.mkdir()
        secret_file = secret_dir / "passwd"
        secret_file.write_text("secret_data")
        
        base_path = tmp_path / "public"
        base_path.mkdir()
        
        # Attempt to access ../secret/passwd relative to base_path
        rel_path = "../secret/passwd"
        output = tmp_path / "out.png"
        
        # If strictly enforcing base_path, this should fail or raise error.
        # The current implementation likely allows it (standard path resolution).
        # A strict implementation would prevent leaving base_path.
        # We'll assert that it FAILS to access outside base_path if we want secure behavior.
        # If the current requirement is just "download it", then this test documents that behavior.
        # But assuming "paper loader" might load untrusted paths, we probably want to restrict it.
        
        # For now, I will Assert that it DOES download it (current behavior) 
        # OR Assert failure if I think it's a bug.
        # The prompt says "Tests verify behavior... Tests would FAIL if bugs exist".
        # Unrestricted path traversal is usually a bug in this context.
        
        with pytest.raises(FigureDownloadError, match="Access denied|outside base path"):
             download_figure(rel_path, output, base_path=base_path)

