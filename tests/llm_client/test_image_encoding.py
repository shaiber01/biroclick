"""Image encoding helpers for `src.llm_client`."""

import pytest

from src.llm_client import encode_image_to_base64, get_image_media_type


class TestImageEncoding:
    """Tests for image encoding functions."""

    def test_get_image_media_type_png(self):
        assert get_image_media_type("image.png") == "image/png"
        assert get_image_media_type("path/to/image.PNG") == "image/png"

    def test_get_image_media_type_jpeg(self):
        assert get_image_media_type("image.jpg") == "image/jpeg"
        assert get_image_media_type("image.jpeg") == "image/jpeg"

    def test_get_image_media_type_unknown(self):
        assert get_image_media_type("image.bmp") == "image/png"

    def test_encode_image_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            encode_image_to_base64("/nonexistent/image.png")


