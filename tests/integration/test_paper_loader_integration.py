"""
Integration tests for the full paper loading workflow.

These tests verify that the paper loader components (markdown parser, downloader,
validator, cost estimator) work together correctly to produce valid PaperInput
objects from raw markdown files and images.

This test suite minimizes mocking to ensure real file system interactions and
path resolutions work as expected.
"""

import json
import shutil
import pytest
from pathlib import Path

from src.paper_loader import (
    load_paper_from_markdown,
    save_paper_input_json,
    load_paper_input,
    ValidationError,
)
from src.paper_loader.config import DEFAULT_DOWNLOAD_CONFIG

# ═══════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_paper_dir(tmp_path):
    """
    Creates a sample directory structure with a markdown paper and images.
    Returns the path to the markdown file.
    """
    paper_dir = tmp_path / "source_paper"
    paper_dir.mkdir()
    
    # Create dummy images
    img_dir = paper_dir / "images"
    img_dir.mkdir()
    
    (img_dir / "fig1.png").write_bytes(b"fake_png_data")
    (img_dir / "fig2.jpg").write_bytes(b"fake_jpg_data")
    
    # Create markdown file
    md_content = """
# Integrated Nanophotonics for Data Centers

## Abstract
We demonstrate a new approach for integrated nanophotonics that significantly improves bandwidth density.
Our method relies on advanced lithography techniques combined with novel material integration.

## Results
The device performance is shown below. We achieved a record-high modulation speed while maintaining low power consumption.
The integration density is 10x higher than previous demonstrations.

![Schematic of the device](images/fig1.png)

The bandwidth measurements indicate high performance across the entire C-band and L-band spectrum.
Signal integrity was maintained up to 100 Gbps per lane.

<img src="images/fig2.jpg" alt="Bandwidth measurements" />

## Conclusion
This works well and scales to future data center needs. The fabrication process is compatible with standard CMOS flows.
    """
    
    md_path = paper_dir / "paper.md"
    md_path.write_text(md_content, encoding="utf-8")
    
    return md_path

@pytest.fixture
def sample_supplementary_dir(tmp_path):
    """
    Creates a sample directory structure for supplementary materials.
    """
    supp_dir = tmp_path / "supplementary"
    supp_dir.mkdir()
    
    (supp_dir / "supp_fig1.png").write_bytes(b"fake_supp_data")
    
    md_content = """
# Supplementary Materials

## Methods
Detailed fabrication steps...

![Fabrication process](supp_fig1.png)
    """
    
    md_path = supp_dir / "supp.md"
    md_path.write_text(md_content, encoding="utf-8")
    
    return md_path

# ═══════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPaperLoaderIntegration:
    """Integration tests for the paper loading workflow."""
    
    def test_full_loading_workflow(self, sample_paper_dir, tmp_path):
        """
        Test the complete flow:
        1. Load from markdown (parse + download figures)
        2. Verify structure and content
        3. Save to JSON
        4. Reload from JSON and verify consistency
        """
        output_dir = tmp_path / "output"
        
        # 1. Load from markdown
        paper_input = load_paper_from_markdown(
            markdown_path=str(sample_paper_dir),
            output_dir=str(output_dir),
            paper_id="test_integration_paper",
            paper_domain="photonics",
            download_figures=True
        )
        
        # 2. Verify structure
        assert paper_input["paper_id"] == "test_integration_paper"
        assert paper_input["paper_title"] == "Integrated Nanophotonics for Data Centers"
        assert "Integrated Nanophotonics" in paper_input["paper_text"]
        assert paper_input["paper_domain"] == "photonics"
        assert len(paper_input["figures"]) == 2
        
        # Verify figures were "downloaded" (copied) correctly
        fig1 = next(f for f in paper_input["figures"] if "fig1" in f["source_url"])
        fig2 = next(f for f in paper_input["figures"] if "fig2" in f["source_url"])
        
        assert Path(fig1["image_path"]).exists()
        assert Path(fig2["image_path"]).exists()
        assert Path(fig1["image_path"]).read_bytes() == b"fake_png_data"
        
        # 3. Save to JSON
        json_path = output_dir / "paper_input.json"
        save_paper_input_json(paper_input, str(json_path))
        assert json_path.exists()
        
        # 4. Reload and verify
        reloaded_input = load_paper_input(str(json_path))
        assert reloaded_input == paper_input
        assert reloaded_input["figures"][0]["id"] == paper_input["figures"][0]["id"]

    def test_loading_with_supplementary(self, sample_paper_dir, sample_supplementary_dir, tmp_path):
        """
        Test loading main paper with supplementary materials.
        Verifies that figure IDs are distinct and correctly prefixed/handled.
        """
        output_dir = tmp_path / "output_supp"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(sample_paper_dir),
            output_dir=str(output_dir),
            paper_id="test_supp",
            supplementary_markdown_path=str(sample_supplementary_dir),
            download_figures=True
        )
        
        # Verify main paper content
        assert len(paper_input["figures"]) == 2
        
        # Verify supplementary content
        assert "supplementary" in paper_input
        supp = paper_input["supplementary"]
        assert "Supplementary Materials" in supp["supplementary_text"]
        assert len(supp["supplementary_figures"]) == 1
        
        # Check ID uniqueness/prefixing
        # Supplementary figures should have 'S' prefix if logic works as expected
        supp_fig_id = supp["supplementary_figures"][0]["id"]
        assert supp_fig_id.startswith("S")
        
        # Verify file exists
        supp_fig_path = supp["supplementary_figures"][0]["image_path"]
        assert Path(supp_fig_path).exists()
        assert Path(supp_fig_path).read_bytes() == b"fake_supp_data"

    def test_error_handling_broken_links(self, sample_paper_dir, tmp_path):
        """
        Test that loader reports download errors but doesn't crash on broken links.
        """
        # Append a broken image link to the markdown
        with open(sample_paper_dir, "a") as f:
            f.write("\n![Broken Image](images/nonexistent.png)")
            
        output_dir = tmp_path / "output_broken"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(sample_paper_dir),
            output_dir=str(output_dir),
            download_figures=True
        )
        
        # Should have processed valid figures
        valid_figs = [f for f in paper_input["figures"] if "download_error" not in f]
        assert len(valid_figs) == 2
        
        # And the broken one
        broken_figs = [f for f in paper_input["figures"] if "download_error" in f]
        assert len(broken_figs) == 1
        assert "Local file not found" in broken_figs[0]["download_error"]

    def test_validation_failure_on_invalid_input(self, tmp_path):
        """
        Test that validation raises error on structurally invalid input 
        (e.g. empty text).
        """
        # Create empty markdown
        md_path = tmp_path / "empty.md"
        md_path.write_text("", encoding="utf-8")
        
        output_dir = tmp_path / "output_invalid"
        
        # Should raise ValidationError because paper text is too short/empty
        with pytest.raises(ValidationError, match="too short"):
            load_paper_from_markdown(
                markdown_path=str(md_path),
                output_dir=str(output_dir)
            )

    def test_path_resolution_security(self, tmp_path):
        """
        Verify that relative paths cannot traverse outside the source directory
        when resolving against base path (implicitly handled by download_figure checks).
        """
        # Setup source dir
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        
        # Create a secret file outside src
        secret = tmp_path / "secret.txt"
        secret.write_text("secret")
        
        # Markdown trying to access secret
        md_path = src_dir / "paper.md"
        # ../secret.txt relative to src/paper.md is tmp_path/secret.txt
        
        # Ensure text is long enough to pass validation (>100 chars)
        filler = "A" * 150
        md_path.write_text(f"# Title\n{filler}\n![Secret](../secret.txt)", encoding="utf-8")
        
        output_dir = tmp_path / "output_sec"
        
        # This should process the markdown, but the download/copy should fail 
        # if we enforced strict base_path checks in previous steps.
        # However, load_paper_from_markdown sets base_path = md_path.parent.
        # download_figure with local file check:
        # "if not local_path.is_absolute() and base_path:"
        #   checks traversal.
        
        # Let's see if it catches it.
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=True
        )
        
        # Check if the figure has a download error
        fig = paper_input["figures"][0]
        assert "download_error" in fig
        assert "Access denied" in fig["download_error"] or "outside base path" in fig["download_error"]


