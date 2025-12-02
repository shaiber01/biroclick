"""
Unit tests for paper_loader validation module.
"""

import pytest
from pathlib import Path

from src.paper_loader import (
    ValidationError,
    validate_paper_input,
    validate_domain,
    validate_figure_image,
    VALID_DOMAINS,
)


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
        """Valid paper input passes validation."""
        paper_input = create_valid_paper_input()
        warnings = validate_paper_input(paper_input)
        # Should return warnings list (possibly empty), not raise
        assert isinstance(warnings, list)
    
    def test_missing_paper_id_raises(self):
        """Missing paper_id raises ValidationError."""
        paper_input = create_valid_paper_input()
        del paper_input["paper_id"]
        
        with pytest.raises(ValidationError, match="Missing required field: paper_id"):
            validate_paper_input(paper_input)
    
    def test_missing_paper_title_raises(self):
        """Missing paper_title raises ValidationError."""
        paper_input = create_valid_paper_input()
        del paper_input["paper_title"]
        
        with pytest.raises(ValidationError, match="Missing required field: paper_title"):
            validate_paper_input(paper_input)
    
    def test_missing_paper_text_raises(self):
        """Missing paper_text raises ValidationError."""
        paper_input = create_valid_paper_input()
        del paper_input["paper_text"]
        
        with pytest.raises(ValidationError, match="Missing required field: paper_text"):
            validate_paper_input(paper_input)
    
    def test_missing_figures_raises(self):
        """Missing figures field raises ValidationError."""
        paper_input = create_valid_paper_input()
        del paper_input["figures"]
        
        with pytest.raises(ValidationError, match="Missing required field: figures"):
            validate_paper_input(paper_input)
    
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
    
    def test_figure_missing_id_raises(self):
        """Figure without id field raises ValidationError."""
        paper_input = create_valid_paper_input()
        paper_input["figures"] = [{"description": "test", "image_path": "test.png"}]
        
        with pytest.raises(ValidationError, match="missing 'id' field"):
            validate_paper_input(paper_input)
    
    def test_figure_missing_image_path_raises(self):
        """Figure without image_path field raises ValidationError."""
        paper_input = create_valid_paper_input()
        paper_input["figures"] = [{"id": "Fig1", "description": "test"}]
        
        with pytest.raises(ValidationError, match="missing 'image_path' field"):
            validate_paper_input(paper_input)
    
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
        # Use an existing file but claim it's a .bmp
        paper_input["figures"] = [{
            "id": "Fig1",
            "description": "test",
            "image_path": str(FIXTURES_DIR / "sample_markdown.md")  # Not an image
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
        warnings = validate_figure_image(image_path)
        # May have other warnings (dimensions, etc.) but not "not found"
        assert not any("Image file not found" in w for w in warnings)
    
    def test_returns_list(self):
        """validate_figure_image always returns a list."""
        warnings = validate_figure_image("/any/path.png")
        assert isinstance(warnings, list)

