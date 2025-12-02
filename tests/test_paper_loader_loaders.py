"""
Integration tests for paper_loader loaders module.
"""

import json
import pytest
from pathlib import Path

from src.paper_loader import (
    load_paper_input,
    create_paper_input,
    load_paper_from_markdown,
    save_paper_input_json,
    get_figure_by_id,
    list_figure_ids,
    get_supplementary_text,
    get_supplementary_figures,
    get_supplementary_data_files,
    get_data_file_by_type,
    get_all_figures,
    ValidationError,
)


# ═══════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "paper_loader"


# ═══════════════════════════════════════════════════════════════════════
# load_paper_input Tests
# ═══════════════════════════════════════════════════════════════════════

class TestLoadPaperInput:
    """Tests for load_paper_input function."""
    
    def test_loads_valid_json(self):
        """Loads valid paper input from JSON file."""
        json_path = FIXTURES_DIR / "valid_paper_input.json"
        paper_input = load_paper_input(str(json_path))
        
        assert paper_input["paper_id"] == "test_valid_paper"
        assert "figures" in paper_input
    
    def test_file_not_found_raises(self):
        """Raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Paper input file not found"):
            load_paper_input("/nonexistent/path.json")
    
    def test_invalid_json_raises_decode_error(self, tmp_path):
        """Raises JSONDecodeError for malformed JSON."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            load_paper_input(str(bad_json))


# ═══════════════════════════════════════════════════════════════════════
# create_paper_input Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCreatePaperInput:
    """Tests for create_paper_input function."""
    
    def test_creates_basic_paper_input(self):
        """Creates basic paper input with required fields."""
        paper_input = create_paper_input(
            paper_id="test_paper",
            paper_title="Test Title",
            paper_text="A" * 150,
            figures=[{"id": "Fig1", "description": "Test", "image_path": "test.png"}]
        )
        
        assert paper_input["paper_id"] == "test_paper"
        assert paper_input["paper_title"] == "Test Title"
        assert len(paper_input["figures"]) == 1
    
    def test_default_domain_is_other(self):
        """Default paper_domain is 'other'."""
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[]
        )
        
        assert paper_input["paper_domain"] == "other"
    
    def test_custom_domain(self):
        """Accepts custom paper_domain."""
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            paper_domain="plasmonics"
        )
        
        assert paper_input["paper_domain"] == "plasmonics"
    
    def test_with_supplementary_text(self):
        """Includes supplementary text."""
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_text="Supplementary methods..."
        )
        
        assert "supplementary" in paper_input
        assert paper_input["supplementary"]["supplementary_text"] == "Supplementary methods..."
    
    def test_with_supplementary_figures(self):
        """Includes supplementary figures."""
        supp_figs = [{"id": "S1", "description": "Supp fig", "image_path": "s1.png"}]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_figures=supp_figs
        )
        
        assert paper_input["supplementary"]["supplementary_figures"] == supp_figs
    
    def test_with_supplementary_data_files(self):
        """Includes supplementary data files."""
        data_files = [{
            "id": "S_data",
            "description": "Data",
            "file_path": "data.csv",
            "data_type": "spectrum"
        }]
        paper_input = create_paper_input(
            paper_id="test",
            paper_title="Test",
            paper_text="A" * 150,
            figures=[],
            supplementary_data_files=data_files
        )
        
        assert paper_input["supplementary"]["supplementary_data_files"] == data_files
    
    def test_validates_on_creation(self):
        """Validates paper input during creation."""
        with pytest.raises(ValidationError):
            create_paper_input(
                paper_id="",  # Invalid: empty
                paper_title="Test",
                paper_text="A" * 150,
                figures=[]
            )


# ═══════════════════════════════════════════════════════════════════════
# load_paper_from_markdown Tests
# ═══════════════════════════════════════════════════════════════════════

class TestLoadPaperFromMarkdown:
    """Tests for load_paper_from_markdown function."""
    
    def test_loads_markdown_extracts_title(self, tmp_path):
        """Extracts paper title from markdown H1."""
        md_path = FIXTURES_DIR / "sample_markdown.md"
        output_dir = tmp_path / "figures"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=False  # Don't try to download
        )
        
        assert "Test Paper Title" in paper_input["paper_title"]
    
    def test_loads_markdown_extracts_figures(self, tmp_path):
        """Extracts figure references from markdown."""
        md_path = FIXTURES_DIR / "sample_markdown.md"
        output_dir = tmp_path / "figures"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=False
        )
        
        # Should find at least the markdown and HTML images
        assert len(paper_input["figures"]) >= 2
    
    def test_generates_paper_id_from_filename(self, tmp_path):
        """Generates paper_id from markdown filename."""
        md_path = FIXTURES_DIR / "sample_markdown.md"
        output_dir = tmp_path / "figures"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=False
        )
        
        assert paper_input["paper_id"] == "sample_markdown"
    
    def test_uses_provided_paper_id(self, tmp_path):
        """Uses provided paper_id instead of filename."""
        md_path = FIXTURES_DIR / "sample_markdown.md"
        output_dir = tmp_path / "figures"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            paper_id="custom_id",
            download_figures=False
        )
        
        assert paper_input["paper_id"] == "custom_id"
    
    def test_sets_paper_domain(self, tmp_path):
        """Sets paper_domain from argument."""
        md_path = FIXTURES_DIR / "sample_markdown.md"
        output_dir = tmp_path / "figures"
        
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            paper_domain="plasmonics",
            download_figures=False
        )
        
        assert paper_input["paper_domain"] == "plasmonics"
    
    def test_file_not_found_raises(self, tmp_path):
        """Raises FileNotFoundError for non-existent markdown."""
        with pytest.raises(FileNotFoundError, match="Markdown file not found"):
            load_paper_from_markdown(
                markdown_path="/nonexistent/paper.md",
                output_dir=str(tmp_path)
            )
    
    def test_creates_output_directory(self, tmp_path):
        """Creates output directory if it doesn't exist."""
        md_path = FIXTURES_DIR / "sample_markdown_noimg.md"
        output_dir = tmp_path / "new" / "nested" / "dir"
        
        load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=False
        )
        
        assert output_dir.exists()
    
    def test_no_images_returns_empty_figures_with_warning(self, tmp_path):
        """Markdown without images returns empty figures list."""
        md_path = FIXTURES_DIR / "sample_markdown_noimg.md"
        output_dir = tmp_path / "figures"
        
        # This will generate a warning about no figures, but shouldn't fail
        paper_input = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=False
        )
        
        assert paper_input["figures"] == []


# ═══════════════════════════════════════════════════════════════════════
# save_paper_input_json Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSavePaperInputJson:
    """Tests for save_paper_input_json function."""
    
    def test_saves_to_json(self, tmp_path):
        """Saves paper input to JSON file."""
        paper_input = {
            "paper_id": "test",
            "paper_title": "Test",
            "paper_text": "A" * 150,
            "figures": []
        }
        output_path = tmp_path / "output.json"
        
        save_paper_input_json(paper_input, str(output_path))
        
        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded["paper_id"] == "test"
    
    def test_creates_parent_directories(self, tmp_path):
        """Creates parent directories if needed."""
        paper_input = {"paper_id": "test", "paper_title": "T", "paper_text": "A"*150, "figures": []}
        output_path = tmp_path / "deep" / "nested" / "output.json"
        
        save_paper_input_json(paper_input, str(output_path))
        
        assert output_path.exists()


# ═══════════════════════════════════════════════════════════════════════
# Accessor Function Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAccessorFunctions:
    """Tests for paper input accessor functions."""
    
    @pytest.fixture
    def sample_paper_input(self):
        """Create a sample paper input for testing accessors."""
        return {
            "paper_id": "test",
            "paper_title": "Test",
            "paper_text": "A" * 150,
            "figures": [
                {"id": "Fig1", "description": "First", "image_path": "fig1.png"},
                {"id": "Fig2", "description": "Second", "image_path": "fig2.png"},
            ],
            "supplementary": {
                "supplementary_text": "Supplementary content",
                "supplementary_figures": [
                    {"id": "S1", "description": "Supp fig", "image_path": "s1.png"}
                ],
                "supplementary_data_files": [
                    {"id": "D1", "description": "Data 1", "file_path": "d1.csv", "data_type": "spectrum"},
                    {"id": "D2", "description": "Data 2", "file_path": "d2.csv", "data_type": "geometry"},
                ]
            }
        }
    
    def test_get_figure_by_id_found(self, sample_paper_input):
        """get_figure_by_id returns figure when found."""
        fig = get_figure_by_id(sample_paper_input, "Fig1")
        
        assert fig is not None
        assert fig["description"] == "First"
    
    def test_get_figure_by_id_not_found(self, sample_paper_input):
        """get_figure_by_id returns None when not found."""
        fig = get_figure_by_id(sample_paper_input, "NonExistent")
        
        assert fig is None
    
    def test_list_figure_ids(self, sample_paper_input):
        """list_figure_ids returns all figure IDs."""
        ids = list_figure_ids(sample_paper_input)
        
        assert ids == ["Fig1", "Fig2"]
    
    def test_get_supplementary_text(self, sample_paper_input):
        """get_supplementary_text returns supplementary text."""
        text = get_supplementary_text(sample_paper_input)
        
        assert text == "Supplementary content"
    
    def test_get_supplementary_text_none(self):
        """get_supplementary_text returns None when not present."""
        paper_input = {"paper_id": "t", "paper_title": "T", "paper_text": "A"*150, "figures": []}
        text = get_supplementary_text(paper_input)
        
        assert text is None
    
    def test_get_supplementary_figures(self, sample_paper_input):
        """get_supplementary_figures returns supplementary figures."""
        figs = get_supplementary_figures(sample_paper_input)
        
        assert len(figs) == 1
        assert figs[0]["id"] == "S1"
    
    def test_get_supplementary_figures_empty(self):
        """get_supplementary_figures returns empty list when none."""
        paper_input = {"paper_id": "t", "paper_title": "T", "paper_text": "A"*150, "figures": []}
        figs = get_supplementary_figures(paper_input)
        
        assert figs == []
    
    def test_get_supplementary_data_files(self, sample_paper_input):
        """get_supplementary_data_files returns data files."""
        files = get_supplementary_data_files(sample_paper_input)
        
        assert len(files) == 2
    
    def test_get_data_file_by_type(self, sample_paper_input):
        """get_data_file_by_type filters by data_type."""
        spectrum_files = get_data_file_by_type(sample_paper_input, "spectrum")
        
        assert len(spectrum_files) == 1
        assert spectrum_files[0]["id"] == "D1"
    
    def test_get_data_file_by_type_no_match(self, sample_paper_input):
        """get_data_file_by_type returns empty for no match."""
        files = get_data_file_by_type(sample_paper_input, "nonexistent_type")
        
        assert files == []
    
    def test_get_all_figures(self, sample_paper_input):
        """get_all_figures returns main + supplementary figures."""
        all_figs = get_all_figures(sample_paper_input)
        
        assert len(all_figs) == 3  # 2 main + 1 supplementary
        ids = [f["id"] for f in all_figs]
        assert "Fig1" in ids
        assert "S1" in ids



