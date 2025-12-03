"""
Integration tests for paper_loader loaders module.
"""

import json
import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, call

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
from src.paper_loader.loaders import _process_figure_refs


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
        assert isinstance(paper_input["figures"], list)
    
    def test_loads_comprehensive_json(self, tmp_path):
        """Loads JSON with all optional fields including supplementary."""
        data = {
            "paper_id": "full_paper",
            "paper_title": "Full Title",
            "paper_text": "A" * 150,
            "paper_domain": "materials",
            "figures": [{"id": "F1", "description": "D1", "image_path": "p1.png"}],
            "supplementary": {
                "supplementary_text": "Supp text",
                "supplementary_figures": [{"id": "S1", "description": "SD1", "image_path": "s1.png"}],
                "supplementary_data_files": [{"id": "D1", "description": "DD1", "file_path": "d1.csv", "data_type": "spectrum"}]
            }
        }
        json_file = tmp_path / "full.json"
        with open(json_file, "w") as f:
            json.dump(data, f)
            
        result = load_paper_input(str(json_file))
        
        assert result == data
        assert result["supplementary"]["supplementary_text"] == "Supp text"
        assert len(result["supplementary"]["supplementary_figures"]) == 1
        assert len(result["supplementary"]["supplementary_data_files"]) == 1

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
            
    @patch("src.paper_loader.loaders.validate_paper_input")
    def test_validates_loaded_json(self, mock_validate, tmp_path):
        """Ensures validate_paper_input is called on loaded data."""
        data = {
            "paper_id": "test",
            "paper_title": "Title",
            "paper_text": "Text",
            "figures": []
        }
        json_file = tmp_path / "test.json"
        with open(json_file, "w") as f:
            json.dump(data, f)
            
        load_paper_input(str(json_file))
        
        mock_validate.assert_called_once_with(data)


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
        assert paper_input["paper_text"] == "A" * 150
        assert len(paper_input["figures"]) == 1
        assert paper_input["figures"][0]["id"] == "Fig1"
        # Default domain
        assert paper_input["paper_domain"] == "other"
        # No supplementary
        assert "supplementary" not in paper_input
    
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
        assert "supplementary_figures" not in paper_input["supplementary"]
    
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
        # We expect ValidationError because ID is empty
        with pytest.raises(ValidationError):
            create_paper_input(
                paper_id="",
                paper_title="Test",
                paper_text="A" * 150,
                figures=[]
            )


# ═══════════════════════════════════════════════════════════════════════
# load_paper_from_markdown Tests
# ═══════════════════════════════════════════════════════════════════════

class TestLoadPaperFromMarkdown:
    """Tests for load_paper_from_markdown function."""
    
    LONG_TEXT = "A" * 150

    @pytest.fixture
    def mock_extract_figures(self):
        with patch("src.paper_loader.loaders.extract_figures_from_markdown") as mock:
            yield mock
            
    @pytest.fixture
    def mock_download(self):
        with patch("src.paper_loader.loaders.download_figure") as mock:
            yield mock
            
    @pytest.fixture
    def mock_check_length(self):
        with patch("src.paper_loader.loaders.check_paper_length") as mock:
            mock.return_value = []
            yield mock

    def test_loads_markdown_basic(self, tmp_path, mock_extract_figures, mock_download, mock_check_length):
        """Tests basic markdown loading."""
        md_path = tmp_path / "test.md"
        text = f"# Title\n{self.LONG_TEXT}"
        md_path.write_text(text, encoding="utf-8")
        output_dir = tmp_path / "figures"
        
        mock_extract_figures.return_value = []
        
        paper = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=False
        )
        
        assert paper["paper_title"] == "Title"
        assert paper["paper_text"] == text
        assert paper["figures"] == []
        assert paper["paper_id"] == "test" # derived from filename

    def test_loads_markdown_extracts_figures(self, tmp_path, mock_extract_figures, mock_download):
        """Extracts figure references from markdown."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(f"{self.LONG_TEXT}\n![Fig 1](fig1.png)", encoding="utf-8")
        output_dir = tmp_path / "figures"
        
        mock_extract_figures.return_value = [
            {"alt": "Fig 1", "url": "fig1.png"}
        ]
        
        with patch("src.paper_loader.loaders.generate_figure_id", return_value="fig1"):
            paper = load_paper_from_markdown(
                markdown_path=str(md_path),
                output_dir=str(output_dir),
                download_figures=True
            )
        
        assert len(paper["figures"]) == 1
        fig = paper["figures"][0]
        assert fig["description"] == "Fig 1"
        assert fig["id"] == "fig1" 
        mock_download.assert_called_once()

    def test_generates_unique_ids_for_duplicates(self, tmp_path, mock_extract_figures, mock_download):
        """Generates unique IDs for duplicate figures."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(self.LONG_TEXT, encoding="utf-8")
        
        # Two figures that would generate same ID
        mock_extract_figures.return_value = [
            {"alt": "Fig 1", "url": "fig.png"},
            {"alt": "Fig 1", "url": "fig.png"}
        ]
        
        with patch("src.paper_loader.loaders.generate_figure_id", return_value="fig1"):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                download_figures=False
            )
        
        assert len(paper["figures"]) == 2
        id1 = paper["figures"][0]["id"]
        id2 = paper["figures"][1]["id"]
        assert id1 == "fig1"
        assert id2 == "fig1_1"

    def test_with_base_url(self, tmp_path, mock_extract_figures, mock_download):
        """Resolves URLs with base_url."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(self.LONG_TEXT, encoding="utf-8")
        
        mock_extract_figures.return_value = [
            {"alt": "Fig", "url": "relative/fig.png"}
        ]
        
        with patch("src.paper_loader.loaders.resolve_figure_url", return_value="http://example.com/relative/fig.png"):
            load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                base_url="http://example.com/",
                download_figures=True
            )
        
        # Check download called with resolved URL
        args, _ = mock_download.call_args
        url = args[0]
        assert url == "http://example.com/relative/fig.png"

    def test_supplementary_markdown(self, tmp_path, mock_extract_figures, mock_download):
        """Loads supplementary markdown and figures."""
        md_path = tmp_path / "main.md"
        md_path.write_text(self.LONG_TEXT, encoding="utf-8")
        supp_path = tmp_path / "supp.md"
        supp_path.write_text("Supp content " + self.LONG_TEXT, encoding="utf-8")
        
        # Configure mock to return different figures for main vs supp calls
        def extract_side_effect(text):
            if text.startswith(self.LONG_TEXT): # Main
                return [{"alt": "Main Fig", "url": "main.png"}]
            if "Supp content" in text: # Supp
                return [{"alt": "Supp Fig", "url": "supp.png"}]
            return []
            
        mock_extract_figures.side_effect = extract_side_effect
        
        with patch("src.paper_loader.loaders.generate_figure_id", side_effect=["main1", "figure_supp"]):
            paper = load_paper_from_markdown(
                markdown_path=str(md_path),
                output_dir=str(tmp_path / "figs"),
                supplementary_markdown_path=str(supp_path),
                download_figures=False
            )
        
        assert paper["supplementary"]["supplementary_text"].startswith("Supp content")
        assert len(paper["figures"]) == 1
        assert len(paper["supplementary"]["supplementary_figures"]) == 1
        
        supp_fig = paper["supplementary"]["supplementary_figures"][0]
        assert supp_fig["id"] == "Sfigure_supp" # Check prefix
        
    def test_download_error_handling(self, tmp_path, mock_extract_figures, mock_download):
        """Handles download errors gracefully."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(self.LONG_TEXT, encoding="utf-8")
        
        mock_extract_figures.return_value = [{"alt": "Fig", "url": "bad.png"}]
        
        # Import exception to mock
        from src.paper_loader.downloader import FigureDownloadError
        mock_download.side_effect = FigureDownloadError("404 Not Found")
        
        paper = load_paper_from_markdown(
            str(md_path),
            str(tmp_path / "figs"),
            download_figures=True
        )
        
        assert len(paper["figures"]) == 1
        assert "download_error" in paper["figures"][0]
        assert "404 Not Found" in paper["figures"][0]["download_error"]

    def test_file_not_found_raises(self, tmp_path):
        """Raises FileNotFoundError for non-existent markdown."""
        with pytest.raises(FileNotFoundError, match="Markdown file not found"):
            load_paper_from_markdown(
                markdown_path="/nonexistent/paper.md",
                output_dir=str(tmp_path)
            )

    def test_no_images_returns_empty_figures_with_warning(self, tmp_path, mock_extract_figures, caplog):
        """Markdown without images returns empty figures list."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(f"No images {self.LONG_TEXT}", encoding="utf-8")
        mock_extract_figures.return_value = []
        
        with caplog.at_level(logging.INFO):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                download_figures=False
            )
            
        assert paper["figures"] == []
        
    def test_output_dir_creation(self, tmp_path, mock_extract_figures):
        """Creates output directory if it doesn't exist."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(self.LONG_TEXT, encoding="utf-8")
        mock_extract_figures.return_value = []
        
        out_dir = tmp_path / "nested" / "dirs"
        assert not out_dir.exists()
        
        load_paper_from_markdown(
            str(md_path),
            str(out_dir),
            download_figures=False
        )
        
        assert out_dir.exists()


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



