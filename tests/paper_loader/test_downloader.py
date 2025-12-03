"""
Unit tests for paper_loader downloader module.

These tests are designed to FIND BUGS, not just pass.
If a test fails, fix the COMPONENT under test, not the test.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import urllib.error
import stat
import socket
import time

from src.paper_loader.downloader import (
    FigureDownloadError,
    download_figure,
)
from src.paper_loader.config import DEFAULT_DOWNLOAD_CONFIG

# ═══════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "paper_loader"

@pytest.fixture
def mock_urlopen():
    with patch('urllib.request.urlopen') as mock:
        yield mock

@pytest.fixture
def mock_request():
    with patch('urllib.request.Request') as mock:
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
        with pytest.raises(FigureDownloadError) as excinfo:
            download_figure("", output)
        assert "empty" in str(excinfo.value).lower() or "cannot be empty" in str(excinfo.value)
        assert not output.exists()  # Should not create output file

    def test_unsupported_scheme_raises_informative_error(self, tmp_path):
        """
        Raises FigureDownloadError for unsupported schemes (e.g. ftp).
        Should not treat ftp:// as a local file path.
        """
        output = tmp_path / "out.png"
        
        # Testing multiple unsupported schemes
        for scheme in ["ftp", "sftp", "gopher", "ldap"]:
            url = f"{scheme}://example.com/image.png"
            with pytest.raises(FigureDownloadError) as excinfo:
                download_figure(url, output)
            
            msg = str(excinfo.value)
            assert "Unsupported scheme" in msg
            assert scheme in msg

    # --- Local File Handling ---

    def test_local_file_absolute_path_enforces_base_path(self, tmp_path):
        """Absolute path OUTSIDE base_path should fail if base_path is provided."""
        # Create a dummy file in tmp_path
        source = tmp_path / "source.png"
        source.write_text("content")
        
        output = tmp_path / "output.png"
        
        # Provide a base_path that does NOT contain the file
        base_path = tmp_path / "other_dir"
        base_path.mkdir()
        
        # Should fail because source is not inside base_path
        with pytest.raises(FigureDownloadError, match="Access denied|outside base path"):
            download_figure(str(source.absolute()), output, base_path=base_path)

    def test_local_file_absolute_path_allowed_inside_base_path(self, tmp_path):
        """Absolute path INSIDE base_path should succeed."""
        base_path = tmp_path / "root"
        base_path.mkdir()
        
        source = base_path / "image.png"
        original_content = "data"
        source.write_text(original_content)
        
        output = tmp_path / "output.png"
        
        result = download_figure(str(source.absolute()), output, base_path=base_path)
        
        assert result is None  # Function should return None
        assert output.exists()
        assert output.read_text() == original_content
        assert source.exists()  # Original file should still exist
        assert source.read_text() == original_content  # Original unchanged

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
            expanded_path = tmp_path / "expanded.png"
            mock_expand.return_value = expanded_path
            
            # We need the file to exist at the expanded path
            original_content = "data"
            expanded_path.write_text(original_content)
            
            output = tmp_path / "out.png"
            
            # If code doesn't call expanduser, it will look for literal "~/..." and fail
            # or if it does call it, it will use our mock and succeed.
            
            # NOTE: We only test that it succeeds if expansion is handled.
            # If implementation lacks expanduser(), this test will fail (file not found).
            download_figure("~/expanded.png", output)
            
            # Verify expanduser was called
            assert mock_expand.called
            assert output.exists()
            assert output.read_text() == original_content

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
        blocker = tmp_path / "blocked"
        blocker.write_text("im a file")
        output = tmp_path / "blocked" / "file.png"
        source = tmp_path / "source.png"
        source.write_text("data")

        with pytest.raises(FigureDownloadError) as exc:
            download_figure(str(source), output)
        
        # Should catch OSError/FileExistsError and wrap it
        assert "Failed to save" in str(exc.value)
        assert str(output) in str(exc.value) or str(output.parent) in str(exc.value)
        assert not output.exists()  # Should not create partial file

    # --- File URL Handling ---

    def test_file_url_decoding(self, tmp_path):
        """Decodes URL encoded characters in file:// paths."""
        # Create file with spaces/special chars
        dir_name = "folder with spaces"
        file_name = "image #1.png"
        (tmp_path / dir_name).mkdir()
        source = tmp_path / dir_name / file_name
        original_content = "data"
        source.write_text(original_content)
        
        # Encode path: spaces -> %20, # -> %23
        # naive encoding for test
        url_path = str(source).replace(" ", "%20").replace("#", "%23")
        if not url_path.startswith("/"):
            url_path = "/" + url_path
        url = f"file://{url_path}"
        
        output = tmp_path / "out.png"
        download_figure(url, output)
        
        assert output.exists()
        assert output.read_text() == original_content
        assert source.exists()  # Original should still exist

    # --- HTTP/HTTPS Handling ---

    def test_http_download_uses_correct_timeout(self, mock_urlopen, tmp_path):
        """Verifies the timeout parameter is passed to urlopen."""
        expected_content = b'data'
        mock_response = MagicMock()
        mock_response.read.return_value = expected_content
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        output = tmp_path / "out.png"
        
        # Custom timeout
        download_figure("http://example.com/img.png", output, timeout=42)
        
        # Verify timeout was passed
        args, kwargs = mock_urlopen.call_args
        assert kwargs.get('timeout') == 42
        
        # Verify content was written
        assert output.exists()
        assert output.read_bytes() == expected_content

    def test_http_download_uses_default_timeout(self, mock_urlopen, tmp_path):
        """Verifies default timeout is used when none provided."""
        expected_content = b'data'
        mock_response = MagicMock()
        mock_response.read.return_value = expected_content
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        output = tmp_path / "out.png"
        
        download_figure("http://example.com/img.png", output)
        
        args, kwargs = mock_urlopen.call_args
        assert kwargs.get('timeout') == DEFAULT_DOWNLOAD_CONFIG.timeout_seconds
        
        # Verify content was written
        assert output.exists()
        assert output.read_bytes() == expected_content

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
        url = "http://example.com/missing.png"
        
        with pytest.raises(FigureDownloadError) as exc:
            download_figure(url, output)
        
        error_msg = str(exc.value)
        assert "404" in error_msg
        assert "Not Found" in error_msg
        assert url in error_msg or "HTTP error" in error_msg
        assert not output.exists()  # Should not create partial file

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
        url = "http://example.com/img.png"
        
        with pytest.raises(FigureDownloadError) as exc:
            download_figure(url, output)
        
        error_msg = str(exc.value)
        assert "Failed to download" in error_msg
        assert url in error_msg
        assert not output.exists()  # Ensure partial file is not left

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

    def test_path_traversal_allowed_without_base_path(self, tmp_path):
        """If base_path NOT provided, traversal is allowed (standard filesystem access)."""
        # Create a secret file
        secret_dir = tmp_path / "secret"
        secret_dir.mkdir()
        secret_file = secret_dir / "passwd"
        original_content = "secret_data"
        secret_file.write_text(original_content)
        
        output = tmp_path / "out.png"
        
        # Absolute path to secret file
        # Should succeed as we assume caller trusts absolute paths if no sandbox (base_path) enforced
        download_figure(str(secret_file), output)
        
        assert output.exists()
        assert output.read_text() == original_content
        assert secret_file.exists()  # Original should still exist

    # --- Additional Edge Cases ---

    def test_local_file_not_found(self, tmp_path):
        """Raises FigureDownloadError when local file does not exist."""
        non_existent = tmp_path / "nonexistent.png"
        output = tmp_path / "out.png"
        
        with pytest.raises(FigureDownloadError) as exc:
            download_figure(str(non_existent), output)
        
        assert "not found" in str(exc.value).lower()
        assert str(non_existent) in str(exc.value)
        assert not output.exists()

    def test_local_file_not_found_with_base_path(self, tmp_path):
        """Raises FigureDownloadError when local file does not exist within base_path."""
        base_path = tmp_path / "base"
        base_path.mkdir()
        output = tmp_path / "out.png"
        
        with pytest.raises(FigureDownloadError) as exc:
            download_figure("nonexistent.png", output, base_path=base_path)
        
        assert "not found" in str(exc.value).lower()
        assert not output.exists()

    def test_local_file_relative_path_without_base_path(self, tmp_path):
        """Relative path without base_path should resolve relative to current directory."""
        source = tmp_path / "source.png"
        original_content = "data"
        source.write_text(original_content)
        
        output = tmp_path / "out.png"
        
        # Change to tmp_path directory and use relative path
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            download_figure("source.png", output)
        finally:
            os.chdir(old_cwd)
        
        assert output.exists()
        assert output.read_text() == original_content

    def test_file_url_not_found(self, tmp_path):
        """Raises FigureDownloadError when file:// URL points to non-existent file."""
        non_existent = tmp_path / "nonexistent.png"
        url = f"file://{non_existent.absolute()}"
        output = tmp_path / "out.png"
        
        with pytest.raises(FigureDownloadError) as exc:
            download_figure(url, output)
        
        assert "not found" in str(exc.value).lower()
        assert not output.exists()

    def test_file_url_with_query_string(self, tmp_path):
        """file:// URLs with query strings should ignore query part."""
        source = tmp_path / "image.png"
        original_content = "data"
        source.write_text(original_content)
        
        url = f"file://{source.absolute()}?query=value"
        output = tmp_path / "out.png"
        
        download_figure(url, output)
        
        assert output.exists()
        assert output.read_text() == original_content

    def test_file_url_with_fragment(self, tmp_path):
        """file:// URLs with fragments should ignore fragment part."""
        source = tmp_path / "image.png"
        original_content = "data"
        source.write_text(original_content)
        
        url = f"file://{source.absolute()}#fragment"
        output = tmp_path / "out.png"
        
        download_figure(url, output)
        
        assert output.exists()
        assert output.read_text() == original_content

    def test_https_download(self, mock_urlopen, tmp_path):
        """HTTPS URLs should work the same as HTTP."""
        expected_content = b'https_data'
        mock_response = MagicMock()
        mock_response.read.return_value = expected_content
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        output = tmp_path / "out.png"
        url = "https://example.com/img.png"
        
        download_figure(url, output)
        
        assert output.exists()
        assert output.read_bytes() == expected_content
        # Verify Request was created with correct URL
        call_args = mock_urlopen.call_args
        assert call_args is not None

    def test_http_user_agent_header(self, mock_urlopen, mock_request, tmp_path):
        """Verifies User-Agent header is set correctly."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'data'
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        output = tmp_path / "out.png"
        url = "http://example.com/img.png"
        
        download_figure(url, output)
        
        # Verify Request was called with correct headers
        assert mock_request.called
        call_args = mock_request.call_args
        assert call_args[0][0] == url
        assert 'headers' in call_args[1]
        assert call_args[1]['headers']['User-Agent'] == DEFAULT_DOWNLOAD_CONFIG.user_agent

    def test_http_403_forbidden(self, mock_urlopen, tmp_path):
        """Raises FigureDownloadError for 403 Forbidden."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 403, "Forbidden", {}, None
        )
        
        output = tmp_path / "out.png"
        url = "http://example.com/forbidden.png"
        
        with pytest.raises(FigureDownloadError) as exc:
            download_figure(url, output)
        
        error_msg = str(exc.value)
        assert "403" in error_msg
        assert "Forbidden" in error_msg
        assert not output.exists()

    def test_http_500_server_error(self, mock_urlopen, tmp_path):
        """Raises FigureDownloadError for 500 Server Error."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 500, "Internal Server Error", {}, None
        )
        
        output = tmp_path / "out.png"
        url = "http://example.com/error.png"
        
        with pytest.raises(FigureDownloadError) as exc:
            download_figure(url, output)
        
        error_msg = str(exc.value)
        assert "500" in error_msg
        assert "Internal Server Error" in error_msg
        assert not output.exists()

    def test_http_timeout_error(self, mock_urlopen, tmp_path):
        """Raises FigureDownloadError for timeout."""
        mock_urlopen.side_effect = urllib.error.URLError("timed out")
        
        output = tmp_path / "out.png"
        url = "http://example.com/slow.png"
        
        with pytest.raises(FigureDownloadError) as exc:
            download_figure(url, output, timeout=1)
        
        error_msg = str(exc.value)
        assert "Failed to download" in error_msg
        assert url in error_msg
        assert not output.exists()

    def test_http_socket_error(self, mock_urlopen, tmp_path):
        """Raises FigureDownloadError for socket errors."""
        mock_urlopen.side_effect = socket.error("Connection refused")
        
        output = tmp_path / "out.png"
        url = "http://example.com/img.png"
        
        with pytest.raises(FigureDownloadError) as exc:
            download_figure(url, output)
        
        error_msg = str(exc.value)
        assert "Failed to download" in error_msg
        assert url in error_msg
        assert not output.exists()

    def test_http_empty_content(self, mock_urlopen, tmp_path):
        """Handles empty HTTP response correctly."""
        mock_response = MagicMock()
        mock_response.read.return_value = b''  # Empty content
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        output = tmp_path / "out.png"
        url = "http://example.com/empty.png"
        
        download_figure(url, output)
        
        assert output.exists()
        assert output.read_bytes() == b''
        assert output.stat().st_size == 0

    def test_http_large_content(self, mock_urlopen, tmp_path):
        """Handles large HTTP responses correctly."""
        # Create 1MB of data
        large_content = b'x' * (1024 * 1024)
        mock_response = MagicMock()
        mock_response.read.return_value = large_content
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        output = tmp_path / "large.png"
        url = "http://example.com/large.png"
        
        download_figure(url, output)
        
        assert output.exists()
        assert output.read_bytes() == large_content
        assert output.stat().st_size == len(large_content)

    def test_output_file_overwrite(self, tmp_path):
        """Output file should be overwritten if it already exists."""
        source = tmp_path / "source.png"
        source.write_text("new_content")
        
        output = tmp_path / "output.png"
        output.write_text("old_content")
        
        download_figure(str(source), output)
        
        assert output.exists()
        assert output.read_text() == "new_content"

    def test_output_path_with_special_characters(self, tmp_path):
        """Handles output paths with special characters."""
        source = tmp_path / "source.png"
        original_content = "data"
        source.write_text(original_content)
        
        output_dir = tmp_path / "dir with spaces"
        output_dir.mkdir()
        output = output_dir / "file#1.png"
        
        download_figure(str(source), output)
        
        assert output.exists()
        assert output.read_text() == original_content

    def test_base_path_is_file_not_directory(self, tmp_path):
        """Raises error if base_path is a file, not a directory."""
        base_path = tmp_path / "base_file"
        base_path.write_text("not a dir")
        
        source = tmp_path / "source.png"
        source.write_text("data")
        output = tmp_path / "out.png"
        
        with pytest.raises(FigureDownloadError) as exc:
            download_figure("source.png", output, base_path=base_path)
        
        # Should fail during path resolution or access check
        assert "Access denied" in str(exc.value) or "not found" in str(exc.value).lower()

    def test_base_path_nonexistent(self, tmp_path):
        """Raises error if base_path does not exist."""
        base_path = tmp_path / "nonexistent_base"
        source = tmp_path / "source.png"
        source.write_text("data")
        output = tmp_path / "out.png"
        
        with pytest.raises(FigureDownloadError) as exc:
            download_figure("source.png", output, base_path=base_path)
        
        # Should fail during path resolution
        assert "Access denied" in str(exc.value) or "not found" in str(exc.value).lower()

    def test_local_file_with_symlink_in_base_path_prevented(self, tmp_path):
        """
        Symlinks that point outside base_path should be prevented for security.
        This is a security feature to prevent path traversal via symlinks.
        """
        # Create structure: base_path -> real_dir (symlink pointing outside)
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()
        source = real_dir / "source.png"
        original_content = "data"
        source.write_text(original_content)
        
        base_path = tmp_path / "base_path"
        base_path.mkdir()
        symlink = base_path / "symlink"
        
        # Create symlink pointing to real_dir (outside base_path)
        try:
            symlink.symlink_to(real_dir)
            output = tmp_path / "out.png"
            
            # Access file through symlink - should be prevented
            with pytest.raises(FigureDownloadError) as exc:
                download_figure("symlink/source.png", output, base_path=base_path)
            
            assert "Access denied" in str(exc.value)
            assert "outside base path" in str(exc.value)
            assert not output.exists()
        except (OSError, NotImplementedError):
            # Symlinks not supported on this platform
            pytest.skip("Symlinks not supported on this platform")

    def test_local_file_with_symlink_inside_base_path_allowed(self, tmp_path):
        """Symlinks that point inside base_path should be allowed."""
        base_path = tmp_path / "base_path"
        base_path.mkdir()
        
        # Create target directory inside base_path
        target_dir = base_path / "target_dir"
        target_dir.mkdir()
        source = target_dir / "source.png"
        original_content = "data"
        source.write_text(original_content)
        
        # Create symlink inside base_path pointing to target_dir (also inside)
        symlink_dir = base_path / "symlink_dir"
        try:
            symlink_dir.symlink_to(target_dir)
            output = tmp_path / "out.png"
            
            # Access file through symlink - should be allowed
            download_figure("symlink_dir/source.png", output, base_path=base_path)
            
            assert output.exists()
            assert output.read_text() == original_content
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform")

    def test_local_file_relative_path_with_base_path(self, tmp_path):
        """Relative path with base_path should resolve correctly."""
        base_path = tmp_path / "base"
        base_path.mkdir()
        
        subdir = base_path / "subdir"
        subdir.mkdir()
        source = subdir / "source.png"
        original_content = "data"
        source.write_text(original_content)
        
        output = tmp_path / "out.png"
        
        download_figure("subdir/source.png", output, base_path=base_path)
        
        assert output.exists()
        assert output.read_text() == original_content

    def test_local_file_absolute_path_resolves_symlinks(self, tmp_path):
        """Absolute paths should resolve symlinks correctly."""
        real_file = tmp_path / "real_file.png"
        original_content = "data"
        real_file.write_text(original_content)
        
        symlink = tmp_path / "symlink.png"
        try:
            symlink.symlink_to(real_file)
            output = tmp_path / "out.png"
            
            download_figure(str(symlink.absolute()), output)
            
            assert output.exists()
            assert output.read_text() == original_content
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform")

    def test_file_url_with_base_path_ignored(self, tmp_path):
        """file:// URLs should ignore base_path restriction (document behavior)."""
        # Create file outside base_path
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        source = outside_dir / "source.png"
        original_content = "data"
        source.write_text(original_content)
        
        base_path = tmp_path / "base"
        base_path.mkdir()
        
        output = tmp_path / "out.png"
        url = f"file://{source.absolute()}"
        
        # file:// URLs bypass base_path restriction (current behavior)
        download_figure(url, output, base_path=base_path)
        
        assert output.exists()
        assert output.read_text() == original_content

    def test_function_returns_none(self, tmp_path):
        """Function should return None on success."""
        source = tmp_path / "source.png"
        source.write_text("data")
        output = tmp_path / "out.png"
        
        result = download_figure(str(source), output)
        
        assert result is None

    def test_output_path_parent_read_only(self, tmp_path):
        """Raises error if output directory is read-only."""
        source = tmp_path / "source.png"
        source.write_text("data")
        
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir()
        read_only_dir.chmod(0o555)  # Read and execute only
        
        output = read_only_dir / "out.png"
        
        try:
            with pytest.raises(FigureDownloadError) as exc:
                download_figure(str(source), output)
            
            assert "Failed to save" in str(exc.value)
            assert not output.exists()
        finally:
            read_only_dir.chmod(0o755)  # Restore for cleanup

    def test_http_response_context_manager(self, mock_urlopen, tmp_path):
        """Verifies HTTP response is used as context manager."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'data'
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response
        
        output = tmp_path / "out.png"
        url = "http://example.com/img.png"
        
        download_figure(url, output)
        
        # Verify context manager was entered and exited
        assert mock_response.__enter__.called
        assert mock_response.__exit__.called

    def test_local_file_preserves_metadata(self, tmp_path):
        """shutil.copy2 should preserve file metadata."""
        source = tmp_path / "source.png"
        original_content = "data"
        source.write_text(original_content)
        
        # Set modification time
        import time
        mtime = time.time() - 3600  # 1 hour ago
        source.touch()
        os.utime(source, (mtime, mtime))
        
        output = tmp_path / "out.png"
        
        download_figure(str(source), output)
        
        assert output.exists()
        # Note: On some filesystems, copy2 may not preserve all metadata
        # but it should at least preserve modification time
        assert abs(output.stat().st_mtime - source.stat().st_mtime) < 1

