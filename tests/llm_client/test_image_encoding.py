"""Image encoding helpers for `src.llm_client`."""

import base64
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.llm_client import (
    create_image_content,
    encode_image_to_base64,
    get_image_media_type,
)

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "paper_loader"
SAMPLE_IMAGE_PATH = FIXTURES_DIR / "sample_images" / "test_figure.png"


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_png_path() -> Path:
    """Return path to a real PNG image for testing."""
    assert SAMPLE_IMAGE_PATH.exists(), f"Test fixture not found: {SAMPLE_IMAGE_PATH}"
    return SAMPLE_IMAGE_PATH


@pytest.fixture
def temp_image_file(tmp_path: Path) -> Path:
    """Create a temporary PNG-like file for testing."""
    # Create a minimal valid PNG file (8-byte signature + minimal IHDR)
    # PNG signature: 137 80 78 71 13 10 26 10
    png_signature = b"\x89PNG\r\n\x1a\n"
    # Minimal IHDR chunk (not a complete PNG, but enough to test encoding)
    ihdr_chunk = b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    
    image_path = tmp_path / "test_image.png"
    image_path.write_bytes(png_signature + ihdr_chunk)
    return image_path


@pytest.fixture
def temp_jpeg_file(tmp_path: Path) -> Path:
    """Create a temporary JPEG-like file for testing."""
    # JPEG files start with FFD8FF
    jpeg_signature = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00"
    
    image_path = tmp_path / "test_image.jpg"
    image_path.write_bytes(jpeg_signature + b"\x00" * 100)
    return image_path


@pytest.fixture
def temp_gif_file(tmp_path: Path) -> Path:
    """Create a temporary GIF-like file for testing."""
    # GIF89a signature
    gif_signature = b"GIF89a"
    
    image_path = tmp_path / "test_image.gif"
    image_path.write_bytes(gif_signature + b"\x00" * 100)
    return image_path


@pytest.fixture
def temp_webp_file(tmp_path: Path) -> Path:
    """Create a temporary WebP-like file for testing."""
    # WEBP starts with RIFF....WEBP
    webp_signature = b"RIFF\x00\x00\x00\x00WEBP"
    
    image_path = tmp_path / "test_image.webp"
    image_path.write_bytes(webp_signature + b"\x00" * 100)
    return image_path


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for get_image_media_type
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetImageMediaType:
    """Tests for get_image_media_type function."""

    def test_png_lowercase(self):
        """PNG extension should return image/png."""
        result = get_image_media_type("image.png")
        assert result == "image/png"

    def test_png_uppercase(self):
        """PNG extension (uppercase) should return image/png."""
        result = get_image_media_type("path/to/image.PNG")
        assert result == "image/png"

    def test_png_mixed_case(self):
        """PNG extension (mixed case) should return image/png."""
        result = get_image_media_type("image.PnG")
        assert result == "image/png"

    def test_jpg_lowercase(self):
        """JPG extension should return image/jpeg."""
        result = get_image_media_type("image.jpg")
        assert result == "image/jpeg"

    def test_jpg_uppercase(self):
        """JPG extension (uppercase) should return image/jpeg."""
        result = get_image_media_type("IMAGE.JPG")
        assert result == "image/jpeg"

    def test_jpeg_lowercase(self):
        """JPEG extension should return image/jpeg."""
        result = get_image_media_type("image.jpeg")
        assert result == "image/jpeg"

    def test_jpeg_uppercase(self):
        """JPEG extension (uppercase) should return image/jpeg."""
        result = get_image_media_type("IMAGE.JPEG")
        assert result == "image/jpeg"

    def test_gif_lowercase(self):
        """GIF extension should return image/gif."""
        result = get_image_media_type("animation.gif")
        assert result == "image/gif"

    def test_gif_uppercase(self):
        """GIF extension (uppercase) should return image/gif."""
        result = get_image_media_type("ANIMATION.GIF")
        assert result == "image/gif"

    def test_webp_lowercase(self):
        """WEBP extension should return image/webp."""
        result = get_image_media_type("photo.webp")
        assert result == "image/webp"

    def test_webp_uppercase(self):
        """WEBP extension (uppercase) should return image/webp."""
        result = get_image_media_type("PHOTO.WEBP")
        assert result == "image/webp"

    def test_unknown_extension_defaults_to_png(self):
        """Unknown extensions should default to image/png."""
        result = get_image_media_type("image.bmp")
        assert result == "image/png"

    def test_tiff_defaults_to_png(self):
        """TIFF (unsupported) should default to image/png."""
        result = get_image_media_type("document.tiff")
        assert result == "image/png"

    def test_svg_defaults_to_png(self):
        """SVG (unsupported) should default to image/png."""
        result = get_image_media_type("vector.svg")
        assert result == "image/png"

    def test_no_extension_defaults_to_png(self):
        """File without extension should default to image/png."""
        result = get_image_media_type("noextension")
        assert result == "image/png"

    def test_empty_string_defaults_to_png(self):
        """Empty string should default to image/png (no extension)."""
        result = get_image_media_type("")
        assert result == "image/png"

    def test_accepts_path_object(self):
        """Should accept Path objects, not just strings."""
        result = get_image_media_type(Path("path/to/image.png"))
        assert result == "image/png"

    def test_path_with_multiple_dots(self):
        """File with multiple dots should use last extension."""
        result = get_image_media_type("image.backup.2023.png")
        assert result == "image/png"

    def test_path_with_dot_directory(self):
        """Path with dot in directory should still extract correct extension."""
        result = get_image_media_type(".hidden/folder.backup/image.jpeg")
        assert result == "image/jpeg"

    def test_hidden_file_with_extension(self):
        """Hidden files (starting with .) should work correctly."""
        result = get_image_media_type(".hidden_image.gif")
        assert result == "image/gif"

    def test_hidden_file_without_extension(self):
        """Hidden file without extension after dot should default to png."""
        result = get_image_media_type(".gitignore")
        assert result == "image/png"

    def test_extension_only(self):
        """File that is just an extension should return correct type."""
        # .png suffix of ".png" is ".png"
        result = get_image_media_type(".png")
        assert result == "image/png"

    def test_whitespace_in_path(self):
        """Path with whitespace should work correctly."""
        result = get_image_media_type("path with spaces/my image.png")
        assert result == "image/png"

    def test_unicode_in_path(self):
        """Path with unicode characters should work correctly."""
        result = get_image_media_type("imágenes/fotos/café.jpg")
        assert result == "image/jpeg"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for encode_image_to_base64
# ═══════════════════════════════════════════════════════════════════════════════


class TestEncodeImageToBase64:
    """Tests for encode_image_to_base64 function."""

    def test_nonexistent_file_raises_error(self):
        """Should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            encode_image_to_base64("/nonexistent/path/image.png")
        
        assert "Image not found" in str(exc_info.value)
        assert "/nonexistent/path/image.png" in str(exc_info.value)

    def test_nonexistent_file_with_path_object(self):
        """Should raise FileNotFoundError for nonexistent Path object."""
        with pytest.raises(FileNotFoundError):
            encode_image_to_base64(Path("/nonexistent/path/image.png"))

    def test_encodes_real_image_file(self, sample_png_path: Path):
        """Should successfully encode a real PNG image."""
        result = encode_image_to_base64(sample_png_path)
        
        # Should return a non-empty string
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Should be valid base64 (no exception when decoding)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_encoded_content_matches_original(self, temp_image_file: Path):
        """Decoded base64 should match original file content."""
        original_content = temp_image_file.read_bytes()
        
        result = encode_image_to_base64(temp_image_file)
        decoded = base64.b64decode(result)
        
        assert decoded == original_content

    def test_accepts_string_path(self, temp_image_file: Path):
        """Should accept string path as well as Path object."""
        result_from_path = encode_image_to_base64(temp_image_file)
        result_from_str = encode_image_to_base64(str(temp_image_file))
        
        assert result_from_path == result_from_str

    def test_encodes_jpeg_file(self, temp_jpeg_file: Path):
        """Should successfully encode JPEG files."""
        result = encode_image_to_base64(temp_jpeg_file)
        
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded == temp_jpeg_file.read_bytes()

    def test_encodes_gif_file(self, temp_gif_file: Path):
        """Should successfully encode GIF files."""
        result = encode_image_to_base64(temp_gif_file)
        
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded == temp_gif_file.read_bytes()

    def test_encodes_webp_file(self, temp_webp_file: Path):
        """Should successfully encode WebP files."""
        result = encode_image_to_base64(temp_webp_file)
        
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded == temp_webp_file.read_bytes()

    def test_empty_file(self, tmp_path: Path):
        """Should handle empty files (returns empty base64 string)."""
        empty_file = tmp_path / "empty.png"
        empty_file.write_bytes(b"")
        
        result = encode_image_to_base64(empty_file)
        
        # Empty file encodes to empty string
        assert result == ""
        assert base64.b64decode(result) == b""

    def test_large_file_encoding(self, tmp_path: Path):
        """Should handle larger files correctly."""
        # Create a 1MB file
        large_content = b"\x00" * (1024 * 1024)
        large_file = tmp_path / "large.png"
        large_file.write_bytes(large_content)
        
        result = encode_image_to_base64(large_file)
        decoded = base64.b64decode(result)
        
        assert decoded == large_content

    def test_directory_raises_error(self, tmp_path: Path):
        """Should handle directory path (not a file) gracefully."""
        # tmp_path is a directory - this should fail
        with pytest.raises((FileNotFoundError, IsADirectoryError, PermissionError)):
            encode_image_to_base64(tmp_path)

    def test_path_with_special_characters(self, tmp_path: Path):
        """Should handle paths with special characters."""
        special_dir = tmp_path / "spëcial çhars & symbols"
        special_dir.mkdir()
        special_file = special_dir / "ímage (1).png"
        special_file.write_bytes(b"test content")
        
        result = encode_image_to_base64(special_file)
        decoded = base64.b64decode(result)
        
        assert decoded == b"test content"

    def test_uses_standard_base64_encoding(self, temp_image_file: Path):
        """Should use standard base64 encoding (not URL-safe variant)."""
        # Create content that produces + and / in standard base64
        # but would be different in URL-safe base64
        content = b"\xfb\xef\xbe"  # Encodes to "++/+" in standard base64
        temp_image_file.write_bytes(content)
        
        result = encode_image_to_base64(temp_image_file)
        
        # Standard base64 uses + and /
        # URL-safe would use - and _
        # Verify it can be decoded with standard decoder
        decoded = base64.standard_b64decode(result)
        assert decoded == content


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for create_image_content
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateImageContent:
    """Tests for create_image_content function."""

    def test_returns_correct_structure(self, temp_image_file: Path):
        """Should return a dictionary with correct structure for LangChain."""
        result = create_image_content(temp_image_file)
        
        assert isinstance(result, dict)
        assert "type" in result
        assert result["type"] == "image_url"
        assert "image_url" in result
        assert isinstance(result["image_url"], dict)
        assert "url" in result["image_url"]
        assert "detail" in result["image_url"]

    def test_default_detail_is_auto(self, temp_image_file: Path):
        """Default detail parameter should be 'auto'."""
        result = create_image_content(temp_image_file)
        
        assert result["image_url"]["detail"] == "auto"

    def test_detail_parameter_low(self, temp_image_file: Path):
        """Should accept 'low' detail parameter."""
        result = create_image_content(temp_image_file, detail="low")
        
        assert result["image_url"]["detail"] == "low"

    def test_detail_parameter_high(self, temp_image_file: Path):
        """Should accept 'high' detail parameter."""
        result = create_image_content(temp_image_file, detail="high")
        
        assert result["image_url"]["detail"] == "high"

    def test_url_format_is_data_uri(self, temp_image_file: Path):
        """URL should be a valid data URI format."""
        result = create_image_content(temp_image_file)
        url = result["image_url"]["url"]
        
        assert url.startswith("data:")
        assert ";base64," in url

    def test_url_contains_correct_media_type_png(self, temp_image_file: Path):
        """URL should contain correct media type for PNG."""
        # temp_image_file has .png extension
        result = create_image_content(temp_image_file)
        url = result["image_url"]["url"]
        
        assert url.startswith("data:image/png;base64,")

    def test_url_contains_correct_media_type_jpeg(self, temp_jpeg_file: Path):
        """URL should contain correct media type for JPEG."""
        result = create_image_content(temp_jpeg_file)
        url = result["image_url"]["url"]
        
        assert url.startswith("data:image/jpeg;base64,")

    def test_url_contains_correct_media_type_gif(self, temp_gif_file: Path):
        """URL should contain correct media type for GIF."""
        result = create_image_content(temp_gif_file)
        url = result["image_url"]["url"]
        
        assert url.startswith("data:image/gif;base64,")

    def test_url_contains_correct_media_type_webp(self, temp_webp_file: Path):
        """URL should contain correct media type for WebP."""
        result = create_image_content(temp_webp_file)
        url = result["image_url"]["url"]
        
        assert url.startswith("data:image/webp;base64,")

    def test_url_base64_content_is_correct(self, temp_image_file: Path):
        """Base64 content in URL should decode to original file content."""
        result = create_image_content(temp_image_file)
        url = result["image_url"]["url"]
        
        # Extract base64 part
        base64_part = url.split(";base64,")[1]
        decoded = base64.b64decode(base64_part)
        
        assert decoded == temp_image_file.read_bytes()

    def test_nonexistent_file_raises_error(self):
        """Should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            create_image_content("/nonexistent/image.png")

    def test_accepts_string_path(self, temp_image_file: Path):
        """Should accept string path."""
        result = create_image_content(str(temp_image_file))
        
        assert result["type"] == "image_url"
        assert "data:image/png;base64," in result["image_url"]["url"]

    def test_accepts_path_object(self, temp_image_file: Path):
        """Should accept Path object."""
        result = create_image_content(temp_image_file)
        
        assert result["type"] == "image_url"
        assert "data:image/png;base64," in result["image_url"]["url"]

    def test_real_image_encoding(self, sample_png_path: Path):
        """Should correctly encode a real PNG image."""
        result = create_image_content(sample_png_path)
        
        # Verify structure
        assert result["type"] == "image_url"
        assert result["image_url"]["detail"] == "auto"
        
        # Verify URL format
        url = result["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        
        # Verify base64 is valid and decodes to original
        base64_part = url.split(";base64,")[1]
        decoded = base64.b64decode(base64_part)
        original = sample_png_path.read_bytes()
        assert decoded == original

    def test_consistent_output_for_same_input(self, temp_image_file: Path):
        """Calling with same input should produce identical output."""
        result1 = create_image_content(temp_image_file)
        result2 = create_image_content(temp_image_file)
        
        assert result1 == result2

    def test_different_files_produce_different_output(
        self, temp_image_file: Path, temp_jpeg_file: Path
    ):
        """Different files should produce different output."""
        result_png = create_image_content(temp_image_file)
        result_jpeg = create_image_content(temp_jpeg_file)
        
        # URLs should be different (different content and media type)
        assert result_png["image_url"]["url"] != result_jpeg["image_url"]["url"]


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestImageEncodingIntegration:
    """Integration tests for image encoding workflow."""

    def test_full_workflow_png(self, sample_png_path: Path):
        """Test complete workflow: media type detection → encoding → content creation."""
        # Step 1: Get media type
        media_type = get_image_media_type(sample_png_path)
        assert media_type == "image/png"
        
        # Step 2: Encode to base64
        base64_str = encode_image_to_base64(sample_png_path)
        assert len(base64_str) > 0
        
        # Step 3: Create content
        content = create_image_content(sample_png_path)
        
        # Verify it all comes together
        expected_url_prefix = f"data:{media_type};base64,"
        assert content["image_url"]["url"].startswith(expected_url_prefix)
        
        # The base64 in content should match direct encoding
        url_base64 = content["image_url"]["url"].split(";base64,")[1]
        assert url_base64 == base64_str

    def test_full_workflow_with_different_extensions(
        self, temp_jpeg_file: Path, temp_gif_file: Path, temp_webp_file: Path
    ):
        """Test workflow works consistently for different image types."""
        test_files = [
            (temp_jpeg_file, "image/jpeg"),
            (temp_gif_file, "image/gif"),
            (temp_webp_file, "image/webp"),
        ]
        
        for file_path, expected_media_type in test_files:
            # Verify media type
            assert get_image_media_type(file_path) == expected_media_type
            
            # Verify encoding works
            base64_str = encode_image_to_base64(file_path)
            decoded = base64.b64decode(base64_str)
            assert decoded == file_path.read_bytes()
            
            # Verify content creation
            content = create_image_content(file_path)
            assert content["image_url"]["url"].startswith(f"data:{expected_media_type};base64,")

    def test_path_string_vs_path_object_consistency(self, temp_image_file: Path):
        """String paths and Path objects should produce identical results."""
        path_str = str(temp_image_file)
        path_obj = temp_image_file
        
        # Media type
        assert get_image_media_type(path_str) == get_image_media_type(path_obj)
        
        # Encoding
        assert encode_image_to_base64(path_str) == encode_image_to_base64(path_obj)
        
        # Content creation
        assert create_image_content(path_str) == create_image_content(path_obj)


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Case Tests  
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case tests for image encoding functions."""

    def test_relative_path_handling(self, tmp_path: Path, monkeypatch):
        """Should handle relative paths correctly."""
        # Create a file
        image_file = tmp_path / "image.png"
        image_file.write_bytes(b"test data")
        
        # Change to tmp_path directory
        monkeypatch.chdir(tmp_path)
        
        # Use relative path
        result = encode_image_to_base64("image.png")
        decoded = base64.b64decode(result)
        assert decoded == b"test data"

    def test_symlink_handling(self, tmp_path: Path):
        """Should follow symlinks correctly."""
        # Create actual file
        actual_file = tmp_path / "actual.png"
        actual_file.write_bytes(b"actual content")
        
        # Create symlink
        symlink = tmp_path / "link.png"
        try:
            symlink.symlink_to(actual_file)
        except OSError:
            pytest.skip("Symlinks not supported on this system")
        
        # Encode via symlink
        result = encode_image_to_base64(symlink)
        decoded = base64.b64decode(result)
        
        assert decoded == b"actual content"

    def test_binary_content_preserved(self, tmp_path: Path):
        """All binary content including null bytes should be preserved."""
        # Create file with various binary patterns including null bytes
        binary_content = bytes(range(256)) * 4
        binary_file = tmp_path / "binary.png"
        binary_file.write_bytes(binary_content)
        
        result = encode_image_to_base64(binary_file)
        decoded = base64.b64decode(result)
        
        assert decoded == binary_content

    def test_very_long_path(self, tmp_path: Path):
        """Should handle paths that are close to system limits."""
        # Create nested directories with long names
        long_dir = tmp_path
        for i in range(10):
            long_dir = long_dir / ("a" * 20)
        long_dir.mkdir(parents=True, exist_ok=True)
        
        long_path_file = long_dir / "image.png"
        long_path_file.write_bytes(b"content")
        
        result = encode_image_to_base64(long_path_file)
        decoded = base64.b64decode(result)
        
        assert decoded == b"content"
