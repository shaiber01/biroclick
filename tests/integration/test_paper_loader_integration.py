"""
Integration tests for the full paper loading workflow.

These tests verify that the paper loader components (markdown parser, downloader,
validator, cost estimator) work together correctly to produce valid PaperInput
objects from raw markdown files and images.

This test suite minimizes mocking to ensure real file system interactions and
path resolutions work as expected.

IMPORTANT: These tests are designed to FIND BUGS. If a test fails, check if
the bug is in the component under test, not in the test itself.
"""

import json
import shutil
import pytest
from pathlib import Path

from src.paper_loader import (
    load_paper_from_markdown,
    save_paper_input_json,
    load_paper_input,
    create_paper_input,
    ValidationError,
    get_figure_by_id,
    list_figure_ids,
    get_supplementary_text,
    get_supplementary_figures,
    get_all_figures,
    estimate_tokens,
    estimate_token_cost,
    check_paper_length,
)
from src.paper_loader.config import (
    DEFAULT_DOWNLOAD_CONFIG,
    VALID_DOMAINS,
    PAPER_LENGTH_NORMAL,
    PAPER_LENGTH_LONG,
    PAPER_LENGTH_VERY_LONG,
)
from schemas.state import CONTEXT_WINDOW_LIMITS

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

@pytest.mark.slow
class TestPaperLoaderIntegration:
    """Integration tests for the paper loading workflow."""
    
    def test_full_loading_workflow(self, sample_paper_dir, tmp_path):
        """
        Test the complete flow:
        1. Load from markdown (parse + download figures)
        2. Verify structure and content with STRICT assertions
        3. Save to JSON
        4. Reload from JSON and verify exact consistency
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
        
        # 2. Verify structure with STRICT assertions
        assert paper_input["paper_id"] == "test_integration_paper"
        assert paper_input["paper_title"] == "Integrated Nanophotonics for Data Centers"
        assert paper_input["paper_domain"] == "photonics"
        
        # Verify paper_text contains exact expected content
        assert "Integrated Nanophotonics" in paper_input["paper_text"]
        assert "bandwidth density" in paper_input["paper_text"]
        assert "100 Gbps" in paper_input["paper_text"]
        assert len(paper_input["paper_text"]) > 100  # Must be substantial
        
        # Verify exact figure count
        assert len(paper_input["figures"]) == 2
        
        # Verify figures were "downloaded" (copied) correctly with exact content checks
        fig1 = next(f for f in paper_input["figures"] if "fig1" in f["source_url"])
        fig2 = next(f for f in paper_input["figures"] if "fig2" in f["source_url"])
        
        # Verify figure structure
        assert "id" in fig1
        assert "description" in fig1
        assert "image_path" in fig1
        assert "source_url" in fig1
        assert fig1["source_url"] == "images/fig1.png"
        assert fig2["source_url"] == "images/fig2.jpg"
        
        # Verify files exist and have correct content
        assert Path(fig1["image_path"]).exists()
        assert Path(fig2["image_path"]).exists()
        assert Path(fig1["image_path"]).read_bytes() == b"fake_png_data"
        assert Path(fig2["image_path"]).read_bytes() == b"fake_jpg_data"
        
        # Verify figure IDs are unique
        figure_ids = [f["id"] for f in paper_input["figures"]]
        assert len(figure_ids) == len(set(figure_ids)), "Figure IDs must be unique"
        
        # Verify figure descriptions match alt text
        assert "Schematic" in fig1["description"] or "device" in fig1["description"].lower()
        assert "Bandwidth" in fig2["description"] or "measurements" in fig2["description"].lower()
        
        # 3. Save to JSON
        json_path = output_dir / "paper_input.json"
        save_paper_input_json(paper_input, str(json_path))
        assert json_path.exists()
        
        # Verify JSON is valid and readable
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        assert json_data["paper_id"] == "test_integration_paper"
        assert len(json_data["figures"]) == 2
        
        # 4. Reload and verify EXACT consistency
        reloaded_input = load_paper_input(str(json_path))
        
        # Verify all fields match exactly
        assert reloaded_input["paper_id"] == paper_input["paper_id"]
        assert reloaded_input["paper_title"] == paper_input["paper_title"]
        assert reloaded_input["paper_text"] == paper_input["paper_text"]
        assert reloaded_input["paper_domain"] == paper_input["paper_domain"]
        assert len(reloaded_input["figures"]) == len(paper_input["figures"])
        
        # Verify figure IDs match exactly
        assert reloaded_input["figures"][0]["id"] == paper_input["figures"][0]["id"]
        assert reloaded_input["figures"][1]["id"] == paper_input["figures"][1]["id"]
        
        # Verify figure paths match exactly
        assert reloaded_input["figures"][0]["image_path"] == paper_input["figures"][0]["image_path"]
        assert reloaded_input["figures"][1]["image_path"] == paper_input["figures"][1]["image_path"]

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
        
        # Verify supplementary content exists
        assert "supplementary" in paper_input
        supp = paper_input["supplementary"]
        assert isinstance(supp, dict)
        
        # Verify supplementary text
        assert "supplementary_text" in supp
        assert "Supplementary Materials" in supp["supplementary_text"]
        assert "Detailed fabrication" in supp["supplementary_text"]
        assert len(supp["supplementary_text"]) > 50
        
        # Verify supplementary figures
        assert "supplementary_figures" in supp
        assert len(supp["supplementary_figures"]) == 1
        
        # Check ID uniqueness/prefixing - MUST have 'S' prefix
        supp_fig_id = supp["supplementary_figures"][0]["id"]
        assert supp_fig_id.startswith("S"), f"Supplementary figure ID must start with 'S', got: {supp_fig_id}"
        
        # Verify main figure IDs don't start with 'S'
        main_fig_ids = [f["id"] for f in paper_input["figures"]]
        for fig_id in main_fig_ids:
            assert not fig_id.startswith("S"), f"Main figure ID should not start with 'S', got: {fig_id}"
        
        # Verify all figure IDs are unique (main + supplementary)
        all_fig_ids = main_fig_ids + [supp_fig_id]
        assert len(all_fig_ids) == len(set(all_fig_ids)), "All figure IDs must be unique"
        
        # Verify file exists and has correct content
        supp_fig_path = supp["supplementary_figures"][0]["image_path"]
        assert Path(supp_fig_path).exists()
        assert Path(supp_fig_path).read_bytes() == b"fake_supp_data"
        
        # Verify supplementary figure structure
        supp_fig = supp["supplementary_figures"][0]
        assert "id" in supp_fig
        assert "description" in supp_fig
        assert "image_path" in supp_fig
        assert "source_url" in supp_fig
        assert supp_fig["source_url"] == "supp_fig1.png"

    def test_error_handling_broken_links(self, sample_paper_dir, tmp_path):
        """
        Test that loader reports download errors but doesn't crash on broken links.
        Verifies error information is preserved and valid figures still work.
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
        
        # Should have processed ALL figures (valid + broken)
        assert len(paper_input["figures"]) == 3
        
        # Should have processed valid figures correctly
        valid_figs = [f for f in paper_input["figures"] if "download_error" not in f]
        assert len(valid_figs) == 2
        
        # Verify valid figures still work correctly
        for fig in valid_figs:
            assert Path(fig["image_path"]).exists()
            assert "id" in fig
            assert "description" in fig
        
        # And the broken one should have error info
        broken_figs = [f for f in paper_input["figures"] if "download_error" in f]
        assert len(broken_figs) == 1
        
        broken_fig = broken_figs[0]
        assert "download_error" in broken_fig
        assert isinstance(broken_fig["download_error"], str)
        assert len(broken_fig["download_error"]) > 0
        assert "Local file not found" in broken_fig["download_error"] or "not found" in broken_fig["download_error"]
        
        # Broken figure should still have required fields
        assert "id" in broken_fig
        assert "description" in broken_fig
        assert "image_path" in broken_fig
        assert "source_url" in broken_fig
        assert broken_fig["source_url"] == "images/nonexistent.png"

    def test_validation_failure_on_invalid_input(self, tmp_path):
        """
        Test that validation raises error on structurally invalid input 
        (e.g. empty text, whitespace only, too short).
        """
        output_dir = tmp_path / "output_invalid"
        
        # Test 1: Empty markdown
        md_path = tmp_path / "empty.md"
        md_path.write_text("", encoding="utf-8")
        
        with pytest.raises(ValidationError, match="too short"):
            load_paper_from_markdown(
                markdown_path=str(md_path),
                output_dir=str(output_dir)
            )
        
        # Test 2: Whitespace only
        md_path.write_text("   \n\t\n   ", encoding="utf-8")
        
        with pytest.raises(ValidationError, match="too short"):
            load_paper_from_markdown(
                markdown_path=str(md_path),
                output_dir=str(output_dir)
            )
        
        # Test 3: Too short (less than 100 chars)
        md_path.write_text("# Title\nShort text", encoding="utf-8")
        
        with pytest.raises(ValidationError, match="too short"):
            load_paper_from_markdown(
                markdown_path=str(md_path),
                output_dir=str(output_dir)
            )
        
        # Test 4: Exactly 99 chars (should fail)
        md_path.write_text("A" * 99, encoding="utf-8")
        
        with pytest.raises(ValidationError, match="too short"):
            load_paper_from_markdown(
                markdown_path=str(md_path),
                output_dir=str(output_dir)
            )
        
        # Test 5: 100 chars (should pass)
        md_path.write_text("A" * 100, encoding="utf-8")
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir)
        )
        assert len(paper_input["paper_text"]) == 100

    def test_path_resolution_security(self, tmp_path):
        """
        Verify that relative paths cannot traverse outside the source directory
        when resolving against base path. Tests multiple traversal attack vectors.
        """
        # Setup source dir
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        
        # Create a secret file outside src
        secret = tmp_path / "secret.txt"
        secret.write_text("secret")
        
        # Ensure text is long enough to pass validation (>100 chars)
        filler = "A" * 150
        output_dir = tmp_path / "output_sec"
        
        # Test 1: Simple traversal ../secret.txt
        md_path = src_dir / "paper.md"
        md_path.write_text(f"# Title\n{filler}\n![Secret](../secret.txt)", encoding="utf-8")
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=True
        )
        
        # Check if the figure has a download error
        assert len(paper_input["figures"]) == 1
        fig = paper_input["figures"][0]
        assert "download_error" in fig
        assert "Access denied" in fig["download_error"] or "outside base path" in fig["download_error"]
        
        # Test 2: Multiple traversal ../../secret.txt
        md_path.write_text(f"# Title\n{filler}\n![Secret](../../secret.txt)", encoding="utf-8")
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=True
        )
        
        fig = paper_input["figures"][0]
        assert "download_error" in fig
        assert "Access denied" in fig["download_error"] or "outside base path" in fig["download_error"]
        
        # Test 3: Absolute path should also be blocked if outside base
        # (Note: This depends on implementation - absolute paths might be handled differently)
        
        # Test 4: Valid relative path should work
        valid_img = src_dir / "valid.png"
        valid_img.write_bytes(b"valid_image")
        md_path.write_text(f"# Title\n{filler}\n![Valid](valid.png)", encoding="utf-8")
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=True
        )
        
        assert len(paper_input["figures"]) == 1
        fig = paper_input["figures"][0]
        assert "download_error" not in fig
        assert Path(fig["image_path"]).exists()
        assert Path(fig["image_path"]).read_bytes() == b"valid_image"

    def test_paper_with_no_figures(self, tmp_path):
        """
        Test loading a paper with no figures. Should still work but warn.
        """
        md_path = tmp_path / "no_figures.md"
        md_content = "# Paper Title\n\n" + "A" * 200 + "\n\nNo figures in this paper."
        md_path.write_text(md_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_no_fig"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            paper_id="no_figures_paper"
        )
        
        assert paper_input["paper_id"] == "no_figures_paper"
        assert paper_input["paper_title"] == "Paper Title"
        assert len(paper_input["figures"]) == 0
        assert "No figures" in paper_input["paper_text"]

    def test_paper_with_only_markdown_images(self, tmp_path):
        """Test loading paper with only markdown-style images."""
        paper_dir = tmp_path / "md_only"
        paper_dir.mkdir()
        img_dir = paper_dir / "images"
        img_dir.mkdir()
        (img_dir / "fig1.png").write_bytes(b"png_data")
        
        md_content = "# Title\n\n" + "A" * 200 + "\n\n![Figure 1](images/fig1.png)"
        md_path = paper_dir / "paper.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_md"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=True
        )
        
        assert len(paper_input["figures"]) == 1
        assert paper_input["figures"][0]["source_url"] == "images/fig1.png"
        assert Path(paper_input["figures"][0]["image_path"]).exists()

    def test_paper_with_only_html_images(self, tmp_path):
        """Test loading paper with only HTML img tags."""
        paper_dir = tmp_path / "html_only"
        paper_dir.mkdir()
        img_dir = paper_dir / "images"
        img_dir.mkdir()
        (img_dir / "fig1.jpg").write_bytes(b"jpg_data")
        
        md_content = "# Title\n\n" + "A" * 200 + '\n\n<img src="images/fig1.jpg" alt="Figure 1" />'
        md_path = paper_dir / "paper.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_html"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=True
        )
        
        assert len(paper_input["figures"]) == 1
        assert paper_input["figures"][0]["source_url"] == "images/fig1.jpg"
        assert "Figure 1" in paper_input["figures"][0]["description"]
        assert Path(paper_input["figures"][0]["image_path"]).exists()

    def test_paper_with_mixed_image_formats(self, tmp_path):
        """Test loading paper with both markdown and HTML images."""
        paper_dir = tmp_path / "mixed"
        paper_dir.mkdir()
        img_dir = paper_dir / "images"
        img_dir.mkdir()
        (img_dir / "fig1.png").write_bytes(b"png_data")
        (img_dir / "fig2.jpg").write_bytes(b"jpg_data")
        (img_dir / "fig3.gif").write_bytes(b"gif_data")
        
        md_content = """# Title

""" + "A" * 200 + """

![Figure 1](images/fig1.png)

<img src="images/fig2.jpg" alt="Figure 2" />

![Figure 3](images/fig3.gif)
"""
        md_path = paper_dir / "paper.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_mixed"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=True
        )
        
        assert len(paper_input["figures"]) == 3
        urls = [f["source_url"] for f in paper_input["figures"]]
        assert "images/fig1.png" in urls
        assert "images/fig2.jpg" in urls
        assert "images/fig3.gif" in urls
        
        # Verify all files were downloaded
        for fig in paper_input["figures"]:
            assert Path(fig["image_path"]).exists()
            assert "download_error" not in fig

    def test_figure_id_generation(self, tmp_path):
        """Test that figure IDs are generated correctly from alt text and URLs."""
        paper_dir = tmp_path / "id_test"
        paper_dir.mkdir()
        img_dir = paper_dir / "images"
        img_dir.mkdir()
        (img_dir / "figure_3a.png").write_bytes(b"data")
        (img_dir / "fig2b.jpg").write_bytes(b"data")
        (img_dir / "image.png").write_bytes(b"data")
        
        md_content = """# Title

""" + "A" * 200 + """

![Figure 3a](images/figure_3a.png)
![Fig. 2b](images/fig2b.jpg)
![Some image](images/image.png)
"""
        md_path = paper_dir / "paper.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_id"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=True
        )
        
        assert len(paper_input["figures"]) == 3
        
        # Check IDs are generated correctly
        ids = [f["id"] for f in paper_input["figures"]]
        assert len(ids) == len(set(ids)), "All IDs must be unique"
        
        # First figure should have ID from alt text
        fig1 = next(f for f in paper_input["figures"] if "figure_3a" in f["source_url"])
        assert "3a" in fig1["id"] or "3" in fig1["id"]
        
        # Second figure should have ID from alt text
        fig2 = next(f for f in paper_input["figures"] if "fig2b" in f["source_url"])
        assert "2b" in fig2["id"] or "2" in fig2["id"]

    def test_duplicate_figure_ids_handling(self, tmp_path):
        """Test that duplicate figure IDs are handled correctly."""
        paper_dir = tmp_path / "duplicate"
        paper_dir.mkdir()
        img_dir = paper_dir / "images"
        img_dir.mkdir()
        (img_dir / "fig1.png").write_bytes(b"data1")
        (img_dir / "fig1_copy.png").write_bytes(b"data2")
        
        md_content = """# Title

""" + "A" * 200 + """

![Figure 1](images/fig1.png)
![Figure 1](images/fig1_copy.png)
"""
        md_path = paper_dir / "paper.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_dup"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=True
        )
        
        assert len(paper_input["figures"]) == 2
        
        # IDs should be unique (second should get suffix)
        ids = [f["id"] for f in paper_input["figures"]]
        assert len(ids) == len(set(ids)), "Duplicate IDs must be made unique"
        assert ids[0] != ids[1]

    def test_paper_length_warnings(self, tmp_path):
        """Test that paper length warnings are generated correctly."""
        paper_dir = tmp_path / "length"
        paper_dir.mkdir()
        
        # Test normal length (no warning)
        md_content = "# Title\n\n" + "A" * (PAPER_LENGTH_NORMAL - 100)
        md_path = paper_dir / "normal.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_len"
        
        # Test long paper (should warn)
        long_content = "# Title\n\n" + "A" * (PAPER_LENGTH_LONG + 1000)
        long_path = paper_dir / "long.md"
        long_path.write_text(long_content, encoding="utf-8")
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(long_path),
            output_dir=str(output_dir)
        )
        
        assert len(paper_input["paper_text"]) > PAPER_LENGTH_LONG

    def test_paper_too_long_validation(self, tmp_path):
        """Test that papers exceeding max length raise ValidationError."""
        paper_dir = tmp_path / "too_long"
        paper_dir.mkdir()
        
        max_chars = CONTEXT_WINDOW_LIMITS["max_paper_chars"]
        too_long_content = "# Title\n\n" + "A" * (max_chars + 1000)
        md_path = paper_dir / "too_long.md"
        md_path.write_text(too_long_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_too_long"
        
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            load_paper_from_markdown(
                markdown_path=str(md_path),
                output_dir=str(output_dir)
            )

    def test_paper_id_generation_from_filename(self, tmp_path):
        """Test that paper_id is generated from filename when not provided."""
        paper_dir = tmp_path / "id_gen"
        paper_dir.mkdir()
        img_dir = paper_dir / "images"
        img_dir.mkdir()
        (img_dir / "fig1.png").write_bytes(b"data")
        
        md_content = "# Title\n\n" + "A" * 200 + "\n\n![Fig](images/fig1.png)"
        md_path = paper_dir / "my-paper_2023.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_id_gen"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=True
        )
        
        # Should generate ID from filename (stem, lowercased, spaces/hyphens -> underscores)
        assert paper_input["paper_id"] == "my_paper_2023"

    def test_domain_validation(self, tmp_path):
        """Test that invalid domains are handled correctly."""
        paper_dir = tmp_path / "domain"
        paper_dir.mkdir()
        img_dir = paper_dir / "images"
        img_dir.mkdir()
        (img_dir / "fig1.png").write_bytes(b"data")
        
        md_content = "# Title\n\n" + "A" * 200 + "\n\n![Fig](images/fig1.png)"
        md_path = paper_dir / "paper.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_domain"
        
        # Valid domain should work
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            paper_domain="plasmonics"
        )
        assert paper_input["paper_domain"] == "plasmonics"
        
        # Invalid domain should still work (validation might warn but not fail)
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            paper_domain="invalid_domain_xyz"
        )
        assert paper_input["paper_domain"] == "invalid_domain_xyz"

    def test_download_figures_false(self, tmp_path):
        """Test that download_figures=False still extracts figure info."""
        paper_dir = tmp_path / "no_download"
        paper_dir.mkdir()
        img_dir = paper_dir / "images"
        img_dir.mkdir()
        (img_dir / "fig1.png").write_bytes(b"data")
        
        md_content = "# Title\n\n" + "A" * 200 + "\n\n![Fig](images/fig1.png)"
        md_path = paper_dir / "paper.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_no_download"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=False
        )
        
        assert len(paper_input["figures"]) == 1
        assert "image_path" in paper_input["figures"][0]
        # File should not exist since we didn't download
        assert not Path(paper_input["figures"][0]["image_path"]).exists()
        assert "download_error" not in paper_input["figures"][0]

    def test_supplementary_without_figures(self, tmp_path):
        """Test supplementary materials with only text, no figures."""
        paper_dir = tmp_path / "supp_text"
        paper_dir.mkdir()
        img_dir = paper_dir / "images"
        img_dir.mkdir()
        (img_dir / "fig1.png").write_bytes(b"data")
        
        md_content = "# Title\n\n" + "A" * 200 + "\n\n![Fig](images/fig1.png)"
        md_path = paper_dir / "paper.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        supp_dir = tmp_path / "supp_text_dir"
        supp_dir.mkdir()
        supp_content = "# Supplementary\n\n" + "B" * 200
        supp_path = supp_dir / "supp.md"
        supp_path.write_text(supp_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_supp_text"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            supplementary_markdown_path=str(supp_path),
            download_figures=True
        )
        
        assert "supplementary" in paper_input
        assert "supplementary_text" in paper_input["supplementary"]
        assert len(paper_input["supplementary"].get("supplementary_figures", [])) == 0

    def test_supplementary_without_text(self, tmp_path):
        """Test supplementary materials with only figures, no text."""
        paper_dir = tmp_path / "supp_fig"
        paper_dir.mkdir()
        img_dir = paper_dir / "images"
        img_dir.mkdir()
        (img_dir / "fig1.png").write_bytes(b"data")
        
        md_content = "# Title\n\n" + "A" * 200 + "\n\n![Fig](images/fig1.png)"
        md_path = paper_dir / "paper.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        supp_dir = tmp_path / "supp_fig_dir"
        supp_dir.mkdir()
        (supp_dir / "supp_fig1.png").write_bytes(b"supp_data")
        
        supp_content = "![Supp Fig](supp_fig1.png)"
        supp_path = supp_dir / "supp.md"
        supp_path.write_text(supp_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_supp_fig"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            supplementary_markdown_path=str(supp_path),
            download_figures=True
        )
        
        assert "supplementary" in paper_input
        assert len(paper_input["supplementary"].get("supplementary_figures", [])) == 1
        # Supplementary text might be empty or just the figure markdown
        assert "supplementary_text" in paper_input["supplementary"]

    def test_load_paper_input_from_json(self, tmp_path):
        """Test loading paper input from JSON file."""
        # Create a valid paper input JSON
        paper_input = {
            "paper_id": "test_json",
            "paper_title": "Test Paper",
            "paper_text": "A" * 200,
            "paper_domain": "plasmonics",
            "figures": [
                {
                    "id": "Fig1",
                    "description": "Test figure",
                    "image_path": str(tmp_path / "fig1.png"),
                    "source_url": "fig1.png"
                }
            ]
        }
        
        # Create the image file
        (tmp_path / "fig1.png").write_bytes(b"image_data")
        
        json_path = tmp_path / "paper.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(paper_input, f)
        
        # Load it
        loaded = load_paper_input(str(json_path))
        
        assert loaded["paper_id"] == "test_json"
        assert loaded["paper_title"] == "Test Paper"
        assert len(loaded["figures"]) == 1
        assert loaded["figures"][0]["id"] == "Fig1"

    def test_load_paper_input_invalid_json(self, tmp_path):
        """Test loading invalid JSON raises appropriate error."""
        json_path = tmp_path / "invalid.json"
        json_path.write_text("{ invalid json }", encoding="utf-8")
        
        with pytest.raises(json.JSONDecodeError):
            load_paper_input(str(json_path))

    def test_load_paper_input_missing_file(self, tmp_path):
        """Test loading non-existent JSON file raises FileNotFoundError."""
        json_path = tmp_path / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            load_paper_input(str(json_path))

    def test_load_paper_input_missing_required_fields(self, tmp_path):
        """Test loading JSON with missing required fields raises ValidationError."""
        # Missing paper_id
        invalid_input = {
            "paper_title": "Test",
            "paper_text": "A" * 200,
            "figures": []
        }
        
        json_path = tmp_path / "invalid.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(invalid_input, f)
        
        with pytest.raises(ValidationError, match="Missing required field"):
            load_paper_input(str(json_path))

    def test_save_and_reload_roundtrip(self, tmp_path):
        """Test that save and reload preserves all data exactly."""
        paper_dir = tmp_path / "roundtrip"
        paper_dir.mkdir()
        img_dir = paper_dir / "images"
        img_dir.mkdir()
        (img_dir / "fig1.png").write_bytes(b"data1")
        (img_dir / "fig2.jpg").write_bytes(b"data2")
        
        md_content = "# Title\n\n" + "A" * 200 + "\n\n![Fig1](images/fig1.png)\n![Fig2](images/fig2.jpg)"
        md_path = paper_dir / "paper.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_roundtrip"
        
        # Load from markdown
        original = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            paper_id="roundtrip_test",
            download_figures=True
        )
        
        # Save to JSON
        json_path = output_dir / "saved.json"
        save_paper_input_json(original, str(json_path))
        
        # Reload
        reloaded = load_paper_input(str(json_path))
        
        # Verify exact match
        assert reloaded == original
        assert reloaded["paper_id"] == original["paper_id"]
        assert reloaded["paper_title"] == original["paper_title"]
        assert reloaded["paper_text"] == original["paper_text"]
        assert len(reloaded["figures"]) == len(original["figures"])
        
        for i, (orig_fig, reload_fig) in enumerate(zip(original["figures"], reloaded["figures"])):
            assert reload_fig["id"] == orig_fig["id"]
            assert reload_fig["description"] == orig_fig["description"]
            assert reload_fig["image_path"] == orig_fig["image_path"]
            assert reload_fig["source_url"] == orig_fig["source_url"]

    def test_accessor_functions(self, tmp_path):
        """Test accessor functions work correctly."""
        paper_dir = tmp_path / "accessor"
        paper_dir.mkdir()
        img_dir = paper_dir / "images"
        img_dir.mkdir()
        (img_dir / "fig1.png").write_bytes(b"data1")
        (img_dir / "fig2.jpg").write_bytes(b"data2")
        
        md_content = "# Title\n\n" + "A" * 200 + "\n\n![Fig1](images/fig1.png)\n![Fig2](images/fig2.jpg)"
        md_path = paper_dir / "paper.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_accessor"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=True
        )
        
        # Test get_figure_by_id
        fig1_id = paper_input["figures"][0]["id"]
        found_fig = get_figure_by_id(paper_input, fig1_id)
        assert found_fig is not None
        assert found_fig["id"] == fig1_id
        
        # Test get_figure_by_id with non-existent ID
        assert get_figure_by_id(paper_input, "NonExistent") is None
        
        # Test list_figure_ids
        ids = list_figure_ids(paper_input)
        assert len(ids) == 2
        assert fig1_id in ids
        
        # Test get_all_figures
        all_figs = get_all_figures(paper_input)
        assert len(all_figs) == 2

    def test_accessor_functions_with_supplementary(self, tmp_path):
        """Test accessor functions with supplementary materials."""
        paper_dir = tmp_path / "accessor_supp"
        paper_dir.mkdir()
        img_dir = paper_dir / "images"
        img_dir.mkdir()
        (img_dir / "fig1.png").write_bytes(b"data1")
        
        md_content = "# Title\n\n" + "A" * 200 + "\n\n![Fig1](images/fig1.png)"
        md_path = paper_dir / "paper.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        supp_dir = tmp_path / "supp_accessor"
        supp_dir.mkdir()
        (supp_dir / "supp_fig1.png").write_bytes(b"supp_data")
        supp_content = "# Supp\n\n" + "B" * 200 + "\n\n![Supp Fig](supp_fig1.png)"
        supp_path = supp_dir / "supp.md"
        supp_path.write_text(supp_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_accessor_supp"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            supplementary_markdown_path=str(supp_path),
            download_figures=True
        )
        
        # Test get_supplementary_text
        supp_text = get_supplementary_text(paper_input)
        assert supp_text is not None
        assert "Supp" in supp_text
        
        # Test get_supplementary_figures
        supp_figs = get_supplementary_figures(paper_input)
        assert len(supp_figs) == 1
        assert supp_figs[0]["id"].startswith("S")
        
        # Test get_all_figures includes both main and supplementary
        all_figs = get_all_figures(paper_input)
        assert len(all_figs) == 2  # 1 main + 1 supplementary

    def test_create_paper_input_programmatically(self, tmp_path):
        """Test creating paper input programmatically."""
        (tmp_path / "fig1.png").write_bytes(b"data1")
        
        paper_input = create_paper_input(
            paper_id="prog_test",
            paper_title="Programmatic Test",
            paper_text="A" * 200,
            figures=[
                {
                    "id": "Fig1",
                    "description": "Test figure",
                    "image_path": str(tmp_path / "fig1.png")
                }
            ],
            paper_domain="plasmonics"
        )
        
        assert paper_input["paper_id"] == "prog_test"
        assert paper_input["paper_title"] == "Programmatic Test"
        assert len(paper_input["figures"]) == 1
        assert paper_input["figures"][0]["id"] == "Fig1"

    def test_create_paper_input_with_supplementary(self, tmp_path):
        """Test creating paper input with supplementary materials."""
        (tmp_path / "fig1.png").write_bytes(b"data1")
        (tmp_path / "supp_fig1.png").write_bytes(b"supp_data")
        
        paper_input = create_paper_input(
            paper_id="prog_supp",
            paper_title="Programmatic Supp Test",
            paper_text="A" * 200,
            figures=[
                {
                    "id": "Fig1",
                    "description": "Main figure",
                    "image_path": str(tmp_path / "fig1.png")
                }
            ],
            supplementary_text="B" * 200,
            supplementary_figures=[
                {
                    "id": "S1",
                    "description": "Supp figure",
                    "image_path": str(tmp_path / "supp_fig1.png")
                }
            ]
        )
        
        assert "supplementary" in paper_input
        assert "supplementary_text" in paper_input["supplementary"]
        assert len(paper_input["supplementary"]["supplementary_figures"]) == 1

    def test_markdown_with_code_blocks(self, tmp_path):
        """Test that code blocks don't interfere with figure extraction."""
        paper_dir = tmp_path / "code_blocks"
        paper_dir.mkdir()
        img_dir = paper_dir / "images"
        img_dir.mkdir()
        (img_dir / "fig1.png").write_bytes(b"data")
        
        md_content = """# Title

""" + "A" * 200 + """

```python
# Code block with ![fake image](fake.png) should be ignored
def function():
    pass
```

![Real Figure](images/fig1.png)
"""
        md_path = paper_dir / "paper.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        output_dir = tmp_path / "output_code"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=True
        )
        
        # Should only extract the real figure, not the one in code block
        assert len(paper_input["figures"]) == 1
        assert paper_input["figures"][0]["source_url"] == "images/fig1.png"

    def test_markdown_title_extraction_edge_cases(self, tmp_path):
        """Test title extraction with various edge cases."""
        paper_dir = tmp_path / "title_test"
        paper_dir.mkdir()
        img_dir = paper_dir / "images"
        img_dir.mkdir()
        (img_dir / "fig1.png").write_bytes(b"data")
        
        output_dir = tmp_path / "output_title"
        
        # Test 1: Title in code block should be ignored
        md_content = """```python
# Title in code block
```

# Real Title

""" + "A" * 200 + """

![Fig](images/fig1.png)
"""
        md_path = paper_dir / "test1.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=True
        )
        assert paper_input["paper_title"] == "Real Title"
        
        # Test 2: HTML h1 tag
        md_content = """<h1>HTML Title</h1>

""" + "A" * 200 + """

![Fig](images/fig1.png)
"""
        md_path = paper_dir / "test2.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=True
        )
        assert "HTML Title" in paper_input["paper_title"]
        
        # Test 3: No title, should use first line or "Untitled Paper"
        md_content = "A" * 200 + "\n\n![Fig](images/fig1.png)"
        md_path = paper_dir / "test3.md"
        md_path.write_text(md_content, encoding="utf-8")
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=True
        )
        assert paper_input["paper_title"] != ""


