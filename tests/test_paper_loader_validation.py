"""
Unit tests for paper_loader validation module.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.paper_loader import (
    ValidationError,
    validate_paper_input,
    validate_domain,
    validate_figure_image,
    VALID_DOMAINS,
)
from schemas.state import CONTEXT_WINDOW_LIMITS


# ═══════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "paper_loader"


def create_valid_paper_input():
    """Create a minimal valid paper input for testing."""
    return {
        "paper_id": "test_paper",
        "paper_title": "Test Paper Title",
        "paper_text": "A" * 150,  # Just over minimum length
        "figures": [
            {
                "id": "Fig1",
                "description": "Test figure",
                "image_path": str(FIXTURES_DIR / "sample_images" / "test_figure.png")
            }
        ]
    }


# ═══════════════════════════════════════════════════════════════════════
# validate_paper_input Tests
# ═══════════════════════════════════════════════════════════════════════

class TestValidatePaperInput:
    """Tests for validate_paper_input function."""
    
    def test_valid_input_passes(self):
        """Valid paper input passes validation with no warnings."""
        paper_input = create_valid_paper_input()
        warnings = validate_paper_input(paper_input)
        # Should return empty warnings list
        assert warnings == [], f"Expected no warnings, got: {warnings}"
    
    def test_missing_required_fields_accumulates_errors(self):
        """Missing multiple required fields raises ValidationError with all missing fields listed."""
        paper_input = {}
        
        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)
        
        error_msg = str(excinfo.value)
        assert "Missing required field: paper_id" in error_msg
        assert "Missing required field: paper_title" in error_msg
        assert "Missing required field: paper_text" in error_msg
        assert "Missing required field: figures" in error_msg

    def test_invalid_types_and_values_accumulates_errors(self):
        """Invalid types and values should be accumulated in the error message."""
        paper_input = create_valid_paper_input()
        # Invalid paper_id type (not string)
        paper_input["paper_id"] = 123
        # Invalid paper_text (too short)
        paper_input["paper_text"] = "Short"
        # Invalid figures type (not list)
        paper_input["figures"] = "not a list"
        
        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)
            
        error_msg = str(excinfo.value)
        # We expect ALL these errors to be reported, not just the first one
        assert "paper_id must be a non-empty string" in error_msg
        assert "paper_text is empty or too short" in error_msg
        assert "figures must be a list" in error_msg
    
    def test_empty_paper_id_raises(self):
        """Empty paper_id raises ValidationError."""
        paper_input = create_valid_paper_input()
        paper_input["paper_id"] = ""
        
        with pytest.raises(ValidationError, match="paper_id must be a non-empty string"):
            validate_paper_input(paper_input)
    
    def test_paper_id_with_spaces_warns(self):
        """Paper ID with spaces generates warning."""
        paper_input = create_valid_paper_input()
        paper_input["paper_id"] = "test paper id"
        
        warnings = validate_paper_input(paper_input)
        assert any("contains spaces" in w for w in warnings)
    
    def test_empty_paper_text_raises(self):
        """Empty paper_text raises ValidationError."""
        paper_input = create_valid_paper_input()
        paper_input["paper_text"] = ""
        
        with pytest.raises(ValidationError, match="paper_text is empty or too short"):
            validate_paper_input(paper_input)
    
    def test_short_paper_text_raises(self):
        """Paper text under 100 chars raises ValidationError."""
        paper_input = create_valid_paper_input()
        paper_input["paper_text"] = "A" * 50  # Only 50 chars
        
        with pytest.raises(ValidationError, match="paper_text is empty or too short"):
            validate_paper_input(paper_input)

    def test_paper_text_too_long_raises(self):
        """Paper text exceeding MAX_PAPER_CHARS raises ValidationError."""
        paper_input = create_valid_paper_input()
        max_chars = CONTEXT_WINDOW_LIMITS["max_paper_chars"]
        # Create text slightly larger than max
        paper_input["paper_text"] = "A" * (max_chars + 1)
        
        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)
        
        assert f"Paper exceeds maximum length ({max_chars:,} chars)" in str(excinfo.value)
    
    def test_figures_not_list_raises(self):
        """Non-list figures field raises ValidationError."""
        paper_input = create_valid_paper_input()
        paper_input["figures"] = "not a list"
        
        with pytest.raises(ValidationError, match="figures must be a list"):
            validate_paper_input(paper_input)
    
    def test_empty_figures_warns(self):
        """Empty figures list generates warning."""
        paper_input = create_valid_paper_input()
        paper_input["figures"] = []
        
        warnings = validate_paper_input(paper_input)
        assert any("No figures provided" in w for w in warnings)
    
    def test_figure_missing_required_fields_accumulates(self):
        """Figure with multiple missing fields should report all of them."""
        paper_input = create_valid_paper_input()
        # Empty dictionary for figure
        paper_input["figures"] = [{}]
        
        with pytest.raises(ValidationError) as excinfo:
            validate_paper_input(paper_input)
            
        error_msg = str(excinfo.value)
        assert "Figure 0: missing 'id' field" in error_msg
        assert "Figure 0 (unknown): missing 'image_path' field" in error_msg
    
    def test_figure_nonexistent_image_warns(self):
        """Figure with non-existent image path generates warning."""
        paper_input = create_valid_paper_input()
        paper_input["figures"] = [{
            "id": "Fig1",
            "description": "test",
            "image_path": "/nonexistent/path/image.png"
        }]
        
        warnings = validate_paper_input(paper_input)
        assert any("image file not found" in w for w in warnings)
    
    def test_figure_unusual_format_warns(self):
        """Figure with unusual image format generates warning."""
        paper_input = create_valid_paper_input()
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.suffix", new_callable=lambda: ".bmp"):  # Property mock is tricky here
             
             # Simpler approach: use an image path that explicitly has .bmp suffix string
             # The code uses Path(path).suffix
             paper_input["figures"] = [{
                "id": "Fig1",
                "description": "test",
                "image_path": "image.bmp"
            }]
             
             warnings = validate_paper_input(paper_input)
             
        assert any("unusual image format" in w for w in warnings)
    
    def test_digitized_data_nonexistent_warns(self):
        """Figure with non-existent digitized data path generates warning."""
        paper_input = create_valid_paper_input()
        paper_input["figures"][0]["digitized_data_path"] = "/nonexistent/data.csv"
        
        warnings = validate_paper_input(paper_input)
        assert any("digitized data file not found" in w for w in warnings)


# ═══════════════════════════════════════════════════════════════════════
# validate_domain Tests
# ═══════════════════════════════════════════════════════════════════════

class TestValidateDomain:
    """Tests for validate_domain function."""
    
    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_valid_domains_pass(self, domain):
        """All valid domains return True."""
        assert validate_domain(domain) is True
    
    def test_invalid_domain_returns_false(self):
        """Invalid domain returns False."""
        assert validate_domain("invalid_domain") is False
    
    def test_empty_domain_returns_false(self):
        """Empty string domain returns False."""
        assert validate_domain("") is False

    def test_none_domain_returns_false(self):
        """None domain returns False (assuming handled or fails)."""
        # Currently implementation uses `domain in VALID_DOMAINS`.
        # if domain is None, `None in ["list", "of", "strings"]` returns False, does not raise.
        assert validate_domain(None) is False


# ═══════════════════════════════════════════════════════════════════════
# validate_figure_image Tests
# ═══════════════════════════════════════════════════════════════════════

class TestValidateFigureImage:
    """Tests for validate_figure_image function."""
    
    def test_nonexistent_image_warns(self):
        """Non-existent image file generates warning."""
        warnings = validate_figure_image("/nonexistent/image.png")
        assert len(warnings) == 1
        assert "Image file not found" in warnings[0]
    
    def test_existing_image_no_critical_warnings(self):
        """Existing image file doesn't generate 'not found' warning."""
        image_path = str(FIXTURES_DIR / "sample_images" / "test_figure.png")
        
        # Mock PIL module and Image class
        mock_img = MagicMock()
        mock_img.size = (1000, 1000)
        mock_img_class = MagicMock()
        mock_img_class.open.return_value = mock_img
        mock_img_class.open.return_value.close = MagicMock()
        
        mock_pil = MagicMock()
        mock_pil.Image = mock_img_class
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.stat") as mock_stat, \
             patch.dict("sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}):
             
            mock_stat.return_value.st_size = 1024 * 1024 # 1MB
            
            warnings = validate_figure_image("dummy.png")
            # Should be empty if everything is good
            assert not warnings

    def test_large_file_warns(self):
        """Large file size generates warning."""
        mock_img = MagicMock()
        mock_img.size = (1000, 1000)
        mock_img_class = MagicMock()
        mock_img_class.open.return_value = mock_img
        mock_img_class.open.return_value.close = MagicMock()
        
        mock_pil = MagicMock()
        mock_pil.Image = mock_img_class

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.stat") as mock_stat, \
             patch.dict("sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}):
             
            mock_stat.return_value.st_size = 50 * 1024 * 1024 
            
            warnings = validate_figure_image("large.png")
            assert any("Large file size" in w for w in warnings)

    def test_low_resolution_warns(self):
        """Low resolution image generates warning."""
        mock_img = MagicMock()
        mock_img.size = (10, 10) # Very small
        mock_img_class = MagicMock()
        mock_img_class.open.return_value = mock_img
        mock_img_class.open.return_value.close = MagicMock()
        
        mock_pil = MagicMock()
        mock_pil.Image = mock_img_class
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.stat") as mock_stat, \
             patch.dict("sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}):
             
            mock_stat.return_value.st_size = 1000
            
            warnings = validate_figure_image("tiny.png")
            assert any("Low resolution" in w for w in warnings)

    def test_high_resolution_warns(self):
        """High resolution image generates warning."""
        mock_img = MagicMock()
        mock_img.size = (10000, 10000) # Very large
        mock_img_class = MagicMock()
        mock_img_class.open.return_value = mock_img
        mock_img_class.open.return_value.close = MagicMock()
        
        mock_pil = MagicMock()
        mock_pil.Image = mock_img_class

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.stat") as mock_stat, \
             patch.dict("sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}):
             
            mock_stat.return_value.st_size = 1000
            
            warnings = validate_figure_image("giant.png")
            assert any("Very high resolution" in w for w in warnings)

    def test_extreme_aspect_ratio_warns(self):
        """Extreme aspect ratio generates warning."""
        mock_img = MagicMock()
        mock_img.size = (100, 5000) # 1:50 ratio
        mock_img_class = MagicMock()
        mock_img_class.open.return_value = mock_img
        mock_img_class.open.return_value.close = MagicMock()
        
        mock_pil = MagicMock()
        mock_pil.Image = mock_img_class

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.stat") as mock_stat, \
             patch.dict("sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}):
             
            mock_stat.return_value.st_size = 1000
            
            warnings = validate_figure_image("long.png")
            assert any("Extreme aspect ratio" in w for w in warnings)

    def test_pil_import_error_handled(self):
        """If PIL is missing, should return warning but not crash."""
        
        # We mock sys.modules to NOT have PIL
        # AND we verify import fails.
        
        with patch.dict("sys.modules", {"PIL": None}):
             # When PIL is mapped to None in sys.modules, import raises ModuleNotFoundError
             
            with patch("pathlib.Path.exists", return_value=True), \
                 patch("pathlib.Path.stat") as mock_stat:
            
                mock_stat.return_value.st_size = 1000
                
                warnings = validate_figure_image("no_pil.png")
                assert any("PIL/Pillow not installed" in w for w in warnings)

    def test_image_analysis_exception_handled(self):
        """Generic exception during image analysis is handled."""
        mock_img_class = MagicMock()
        mock_img_class.open.side_effect = Exception("Corrupt image")
        
        mock_pil = MagicMock()
        mock_pil.Image = mock_img_class
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.stat") as mock_stat, \
             patch.dict("sys.modules", {"PIL": mock_pil, "PIL.Image": mock_img_class}):
             
             # Set a valid size to pass size check
             mock_stat.return_value.st_size = 1024
             
             warnings = validate_figure_image("corrupt.png")
             assert any("Could not analyze image" in w for w in warnings)
