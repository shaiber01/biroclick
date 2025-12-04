"""Tests for loading paper inputs from markdown sources."""

import logging
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.paper_loader import load_paper_from_markdown
from src.paper_loader.downloader import FigureDownloadError
from src.paper_loader.validation import ValidationError


LONG_TEXT = "A" * 150


@pytest.fixture
def mock_extract_figures():
    with patch("src.paper_loader.loaders.extract_figures_from_markdown") as mock:
        yield mock


@pytest.fixture
def mock_download():
    with patch("src.paper_loader.loaders.download_figure") as mock:
        yield mock


@pytest.fixture
def mock_check_length():
    with patch("src.paper_loader.loaders.check_paper_length") as mock:
        mock.return_value = []
        yield mock


class TestLoadPaperFromMarkdown:
    """Tests for load_paper_from_markdown function."""

    def test_loads_markdown_basic(
        self, tmp_path, mock_extract_figures, mock_download, mock_check_length
    ):
        """Tests basic markdown loading with comprehensive assertions."""
        md_path = tmp_path / "test.md"
        text = f"# Title\n{LONG_TEXT}"
        md_path.write_text(text, encoding="utf-8")
        output_dir = tmp_path / "figures"

        mock_extract_figures.return_value = []

        paper = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(output_dir),
            download_figures=False,
        )

        # Verify all required fields are present
        assert "paper_id" in paper
        assert "paper_title" in paper
        assert "paper_text" in paper
        assert "paper_domain" in paper
        assert "figures" in paper
        
        # Verify exact values
        assert paper["paper_title"] == "Title"
        assert paper["paper_text"] == text
        assert paper["figures"] == []
        assert paper["paper_id"] == "test"
        assert paper["paper_domain"] == "other"
        
        # Verify supplementary section is not present when not provided
        assert "supplementary" not in paper
        
        # Verify output directory structure was created: {output_dir}/{paper_id}/run_{timestamp}/figures/
        # Since timestamp is dynamic, we check for the run_ prefix pattern
        paper_dir = Path(output_dir) / "test"
        assert paper_dir.exists()
        run_dirs = list(paper_dir.glob("run_*"))
        assert len(run_dirs) == 1
        figures_dir = run_dirs[0] / "figures"
        assert figures_dir.exists()
        
        # Verify run_output_dir is returned
        assert "run_output_dir" in paper
        assert paper["run_output_dir"] == str(run_dirs[0])
        
        # Verify extract_figures was called with correct text
        mock_extract_figures.assert_called_once_with(text)
        
        # Verify download was not called when download_figures=False
        mock_download.assert_not_called()

    def test_loads_markdown_extracts_figures(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Extracts figure references from markdown with full field validation."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(f"{LONG_TEXT}\n![Fig 1](fig1.png)", encoding="utf-8")
        output_dir = tmp_path / "figures"

        mock_extract_figures.return_value = [{"alt": "Fig 1", "url": "fig1.png"}]

        with patch("src.paper_loader.loaders.generate_figure_id", return_value="fig1"):
            paper = load_paper_from_markdown(
                markdown_path=str(md_path),
                output_dir=str(output_dir),
                download_figures=True,
            )

        assert len(paper["figures"]) == 1
        fig = paper["figures"][0]
        
        # Verify all figure fields are present
        assert "id" in fig
        assert "description" in fig
        assert "image_path" in fig
        assert "source_url" in fig
        
        # Verify exact values
        assert fig["description"] == "Fig 1"
        assert fig["id"] == "fig1"
        assert fig["source_url"] == "fig1.png"
        # Figures are now saved under {output_dir}/{paper_id}/run_{timestamp}/figures/
        # Get the run_output_dir from the paper and verify the image_path uses it
        run_output_dir = paper["run_output_dir"]
        expected_path = Path(run_output_dir) / "figures" / "fig1.png"
        assert fig["image_path"] == str(expected_path)
        
        # Verify download was called with correct arguments
        mock_download.assert_called_once()
        call_args = mock_download.call_args
        assert call_args[0][0] == str(Path(md_path.parent) / "fig1.png")  # resolved URL
        assert call_args[0][1] == expected_path  # output path
        assert call_args[1]["timeout"] == 30  # default timeout

    def test_generates_unique_ids_for_duplicates(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Generates unique IDs for duplicate figures with comprehensive validation."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")

        mock_extract_figures.return_value = [
            {"alt": "Fig 1", "url": "fig.png"},
            {"alt": "Fig 1", "url": "fig.png"},
        ]

        with patch("src.paper_loader.loaders.generate_figure_id", return_value="fig1"):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                download_figures=False,
            )

        assert len(paper["figures"]) == 2
        
        # Verify IDs are unique
        id1 = paper["figures"][0]["id"]
        id2 = paper["figures"][1]["id"]
        assert id1 == "fig1"
        assert id2 == "fig1_1"
        assert id1 != id2
        
        # Verify both figures have all required fields
        for fig in paper["figures"]:
            assert "id" in fig
            assert "description" in fig
            assert "image_path" in fig
            assert "source_url" in fig
            assert fig["source_url"] == "fig.png"

    def test_generates_unique_ids_for_many_duplicates(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Generates unique IDs for many duplicate figures."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")

        mock_extract_figures.return_value = [
            {"alt": "Fig 1", "url": "fig.png"}
        ] * 5  # 5 identical figures

        with patch("src.paper_loader.loaders.generate_figure_id", return_value="fig1"):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                download_figures=False,
            )

        assert len(paper["figures"]) == 5
        
        # Verify all IDs are unique
        ids = [fig["id"] for fig in paper["figures"]]
        assert len(ids) == len(set(ids)), "All IDs must be unique"
        assert ids == ["fig1", "fig1_1", "fig1_2", "fig1_3", "fig1_4"]

    def test_with_base_url(self, tmp_path, mock_extract_figures, mock_download):
        """Resolves URLs with base_url and verifies all call arguments."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")

        mock_extract_figures.return_value = [{"alt": "Fig", "url": "relative/fig.png"}]

        with patch(
            "src.paper_loader.loaders.resolve_figure_url",
            return_value="http://example.com/relative/fig.png",
        ):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                base_url="http://example.com/",
                download_figures=True,
            )

        # Verify download was called with resolved URL
        mock_download.assert_called_once()
        args, kwargs = mock_download.call_args
        assert args[0] == "http://example.com/relative/fig.png"
        # Actual ID generation produces "Fig1" (capital F) from alt text "Fig"
        # Figures are now saved under {output_dir}/{paper_id}/run_{timestamp}/figures/
        run_output_dir = paper["run_output_dir"]
        expected_path = Path(run_output_dir) / "figures" / "Fig1.png"
        assert args[1] == expected_path
        assert kwargs["timeout"] == 30
        assert kwargs["base_path"] == md_path.parent
        
        # Verify figure entry has correct source_url
        assert paper["figures"][0]["source_url"] == "relative/fig.png"
        assert paper["figures"][0]["id"] == "Fig1"

    def test_supplementary_markdown(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Loads supplementary markdown and figures with full validation."""
        md_path = tmp_path / "main.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        supp_path = tmp_path / "supp.md"
        supp_text = "Supp content " + LONG_TEXT
        supp_path.write_text(supp_text, encoding="utf-8")

        def extract_side_effect(text):
            if text.startswith(LONG_TEXT):
                return [{"alt": "Main Fig", "url": "main.png"}]
            if "Supp content" in text:
                return [{"alt": "Supp Fig", "url": "supp.png"}]
            return []

        mock_extract_figures.side_effect = extract_side_effect

        with patch(
            "src.paper_loader.loaders.generate_figure_id",
            side_effect=["main1", "figure_supp"],
        ):
            paper = load_paper_from_markdown(
                markdown_path=str(md_path),
                output_dir=str(tmp_path / "figs"),
                supplementary_markdown_path=str(supp_path),
                download_figures=False,
            )

        # Verify supplementary section exists
        assert "supplementary" in paper
        assert "supplementary_text" in paper["supplementary"]
        assert "supplementary_figures" in paper["supplementary"]
        
        # Verify supplementary text content
        assert paper["supplementary"]["supplementary_text"] == supp_text
        assert paper["supplementary"]["supplementary_text"].startswith("Supp content")
        
        # Verify main figures
        assert len(paper["figures"]) == 1
        assert paper["figures"][0]["id"] == "main1"
        assert paper["figures"][0]["description"] == "Main Fig"
        
        # Verify supplementary figures
        assert len(paper["supplementary"]["supplementary_figures"]) == 1
        supp_fig = paper["supplementary"]["supplementary_figures"][0]
        assert supp_fig["id"] == "Sfigure_supp"
        assert supp_fig["description"] == "Supp Fig"
        assert supp_fig["source_url"] == "supp.png"
        
        # Verify supplementary figure has all required fields
        # Figures are now saved under {output_dir}/{paper_id}/run_{timestamp}/figures/
        assert "image_path" in supp_fig
        run_output_dir = paper["run_output_dir"]
        expected_path = Path(run_output_dir) / "figures" / "Sfigure_supp.png"
        assert supp_fig["image_path"] == str(expected_path)

    def test_supplementary_markdown_nonexistent_file(
        self, tmp_path, mock_extract_figures, mock_download, caplog
    ):
        """Handles nonexistent supplementary file gracefully."""
        md_path = tmp_path / "main.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        supp_path = tmp_path / "nonexistent.md"

        mock_extract_figures.return_value = []

        with caplog.at_level(logging.WARNING):
            paper = load_paper_from_markdown(
                markdown_path=str(md_path),
                output_dir=str(tmp_path / "figs"),
                supplementary_markdown_path=str(supp_path),
                download_figures=False,
            )

        # Verify paper loads successfully without supplementary
        assert "supplementary" not in paper
        assert len(paper["figures"]) == 0
        
        # Verify warning was logged
        assert any("Supplementary file not found" in record.message for record in caplog.records)

    def test_supplementary_markdown_empty_file(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Handles empty supplementary file."""
        md_path = tmp_path / "main.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        supp_path = tmp_path / "supp.md"
        supp_path.write_text("", encoding="utf-8")

        mock_extract_figures.side_effect = [[], []]  # No figures in either

        paper = load_paper_from_markdown(
            markdown_path=str(md_path),
            output_dir=str(tmp_path / "figs"),
            supplementary_markdown_path=str(supp_path),
            download_figures=False,
        )

        # Verify supplementary section exists with empty text
        assert "supplementary" in paper
        assert paper["supplementary"]["supplementary_text"] == ""
        assert paper["supplementary"].get("supplementary_figures", []) == []

    def test_download_error_handling(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Handles download errors gracefully with full error field validation."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")

        mock_extract_figures.return_value = [{"alt": "Fig", "url": "bad.png"}]

        mock_download.side_effect = FigureDownloadError("404 Not Found")

        paper = load_paper_from_markdown(
            str(md_path),
            str(tmp_path / "figs"),
            download_figures=True,
        )

        assert len(paper["figures"]) == 1
        fig = paper["figures"][0]
        
        # Verify error field is present and contains error message
        assert "download_error" in fig
        assert isinstance(fig["download_error"], str)
        assert "404 Not Found" in fig["download_error"]
        
        # Verify figure still has all other required fields
        assert "id" in fig
        assert "description" in fig
        assert "image_path" in fig
        assert "source_url" in fig
        assert fig["source_url"] == "bad.png"

    def test_multiple_download_errors(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Handles multiple download errors correctly."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")

        mock_extract_figures.return_value = [
            {"alt": "Fig 1", "url": "bad1.png"},
            {"alt": "Fig 2", "url": "bad2.png"},
            {"alt": "Fig 3", "url": "good.png"},
        ]

        def download_side_effect(url, path, **kwargs):
            if "bad" in url:
                raise FigureDownloadError(f"Failed: {url}")
            # good.png succeeds

        mock_download.side_effect = download_side_effect

        with patch("src.paper_loader.loaders.generate_figure_id", side_effect=["fig1", "fig2", "fig3"]):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                download_figures=True,
            )

        assert len(paper["figures"]) == 3
        
        # Verify errors are recorded correctly
        assert "download_error" in paper["figures"][0]
        assert "download_error" in paper["figures"][1]
        assert "download_error" not in paper["figures"][2]  # good.png succeeded

    def test_file_not_found_raises(self, tmp_path):
        """Raises FileNotFoundError for non-existent markdown with correct message."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_paper_from_markdown(
                markdown_path="/nonexistent/paper.md", output_dir=str(tmp_path)
            )
        
        assert "Markdown file not found" in str(exc_info.value)
        assert "/nonexistent/paper.md" in str(exc_info.value)

    def test_file_not_found_empty_path(self, tmp_path):
        """Raises FileNotFoundError for empty path."""
        with pytest.raises(FileNotFoundError):
            load_paper_from_markdown(
                markdown_path="", output_dir=str(tmp_path)
            )

    def test_no_images_returns_empty_figures_with_warning(
        self, tmp_path, mock_extract_figures, caplog
    ):
        """Markdown without images returns empty figures list."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(f"No images {LONG_TEXT}", encoding="utf-8")
        mock_extract_figures.return_value = []

        with caplog.at_level(logging.INFO):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                download_figures=False,
            )

        assert paper["figures"] == []
        assert isinstance(paper["figures"], list)
        assert len(paper["figures"]) == 0

    def test_output_dir_creation(self, tmp_path, mock_extract_figures):
        """Creates output directory structure if it doesn't exist."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        mock_extract_figures.return_value = []

        out_dir = tmp_path / "nested" / "dirs" / "deep"
        assert not out_dir.exists()

        paper = load_paper_from_markdown(
            str(md_path),
            str(out_dir),
            download_figures=False,
        )

        # Output structure is now {output_dir}/{paper_id}/run_{timestamp}/figures/
        paper_dir = out_dir / "paper"
        assert paper_dir.exists()
        run_dirs = list(paper_dir.glob("run_*"))
        assert len(run_dirs) == 1
        expected_figures_dir = run_dirs[0] / "figures"
        assert expected_figures_dir.exists()
        assert expected_figures_dir.is_dir()
        
        # Verify paper still loads correctly
        assert paper["paper_id"] == "paper"
        assert paper["run_output_dir"] == str(run_dirs[0])

    def test_output_dir_already_exists(self, tmp_path, mock_extract_figures):
        """Handles existing output directory correctly."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        mock_extract_figures.return_value = []

        out_dir = tmp_path / "existing"
        out_dir.mkdir()
        assert out_dir.exists()

        paper = load_paper_from_markdown(
            str(md_path),
            str(out_dir),
            download_figures=False,
        )

        assert out_dir.exists()
        assert paper["paper_id"] == "paper"

    def test_paper_id_from_filename(self, tmp_path, mock_extract_figures):
        """Generates paper_id from filename when not provided."""
        md_path = tmp_path / "my_paper-2023.md"
        md_path.write_text(f"# Title\n{LONG_TEXT}", encoding="utf-8")
        mock_extract_figures.return_value = []

        paper = load_paper_from_markdown(
            str(md_path),
            str(tmp_path / "figs"),
            download_figures=False,
        )

        assert paper["paper_id"] == "my_paper_2023"

    def test_paper_id_explicit(self, tmp_path, mock_extract_figures):
        """Uses explicit paper_id when provided."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(f"# Title\n{LONG_TEXT}", encoding="utf-8")
        mock_extract_figures.return_value = []

        paper = load_paper_from_markdown(
            str(md_path),
            str(tmp_path / "figs"),
            paper_id="custom_id_123",
            download_figures=False,
        )

        assert paper["paper_id"] == "custom_id_123"

    def test_paper_domain_default(self, tmp_path, mock_extract_figures):
        """Uses default paper_domain when not provided."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(f"# Title\n{LONG_TEXT}", encoding="utf-8")
        mock_extract_figures.return_value = []

        paper = load_paper_from_markdown(
            str(md_path),
            str(tmp_path / "figs"),
            download_figures=False,
        )

        assert paper["paper_domain"] == "other"

    def test_paper_domain_explicit(self, tmp_path, mock_extract_figures):
        """Uses explicit paper_domain when provided."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(f"# Title\n{LONG_TEXT}", encoding="utf-8")
        mock_extract_figures.return_value = []

        paper = load_paper_from_markdown(
            str(md_path),
            str(tmp_path / "figs"),
            paper_domain="plasmonics",
            download_figures=False,
        )

        assert paper["paper_domain"] == "plasmonics"

    def test_figure_timeout_custom(self, tmp_path, mock_extract_figures, mock_download):
        """Uses custom figure_timeout when provided."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        mock_extract_figures.return_value = [{"alt": "Fig", "url": "fig.png"}]

        with patch("src.paper_loader.loaders.generate_figure_id", return_value="fig1"):
            load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                download_figures=True,
                figure_timeout=60,
            )

        mock_download.assert_called_once()
        call_args = mock_download.call_args
        assert call_args[1]["timeout"] == 60

    def test_figure_without_alt_text(self, tmp_path, mock_extract_figures, mock_download):
        """Handles figures without alt text correctly."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        mock_extract_figures.return_value = [{"alt": "", "url": "fig.png"}]

        with patch("src.paper_loader.loaders.generate_figure_id", return_value="fig1"):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                download_figures=False,
            )

        assert len(paper["figures"]) == 1
        fig = paper["figures"][0]
        assert fig["description"] == "Figure from paper"  # default description
        assert fig["source_url"] == "fig.png"

    def test_figure_format_warning_non_preferred(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Adds format warning for non-preferred formats."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        mock_extract_figures.return_value = [{"alt": "Fig", "url": "fig.eps"}]

        with patch("src.paper_loader.loaders.generate_figure_id", return_value="fig1"):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                download_figures=False,
            )

        assert len(paper["figures"]) == 1
        fig = paper["figures"][0]
        assert "format_warning" in fig
        assert "eps" in fig["format_warning"].lower()
        assert "preferred" in fig["format_warning"].lower()

    def test_figure_format_no_warning_preferred(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Does not add format warning for preferred formats."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        mock_extract_figures.return_value = [{"alt": "Fig", "url": "fig.png"}]

        with patch("src.paper_loader.loaders.generate_figure_id", return_value="fig1"):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                download_figures=False,
            )

        assert len(paper["figures"]) == 1
        fig = paper["figures"][0]
        assert "format_warning" not in fig

    def test_empty_markdown_file(self, tmp_path, mock_extract_figures):
        """Handles empty markdown file - should raise validation error."""
        md_path = tmp_path / "empty.md"
        md_path.write_text("", encoding="utf-8")
        mock_extract_figures.return_value = []

        # Empty markdown file should fail validation because paper_text is too short
        from src.paper_loader.validation import ValidationError
        with pytest.raises(ValidationError, match="paper_text is empty or too short"):
            load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                download_figures=False,
            )

    def test_markdown_no_title(self, tmp_path, mock_extract_figures):
        """Handles markdown without title heading."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(f"Some content without title\n{LONG_TEXT}", encoding="utf-8")
        mock_extract_figures.return_value = []

        paper = load_paper_from_markdown(
            str(md_path),
            str(tmp_path / "figs"),
            download_figures=False,
        )

        # Should use first non-empty line or "Untitled Paper"
        assert paper["paper_title"] in ["Some content without title", "Untitled Paper"]
        assert paper["paper_text"] == md_path.read_text(encoding="utf-8")

    def test_markdown_multiple_h1_headings(self, tmp_path, mock_extract_figures):
        """Uses first H1 heading as title."""
        md_path = tmp_path / "paper.md"
        text = f"# First Title\n{LONG_TEXT}\n# Second Title\nMore text"
        md_path.write_text(text, encoding="utf-8")
        mock_extract_figures.return_value = []

        paper = load_paper_from_markdown(
            str(md_path),
            str(tmp_path / "figs"),
            download_figures=False,
        )

        assert paper["paper_title"] == "First Title"
        assert paper["paper_text"] == text

    def test_supplementary_base_url_overrides_base_url(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Uses supplementary_base_url when provided, otherwise falls back to base_url."""
        md_path = tmp_path / "main.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        supp_path = tmp_path / "supp.md"
        supp_path.write_text("Supp content", encoding="utf-8")

        mock_extract_figures.side_effect = [
            [],  # main has no figures
            [{"alt": "Supp Fig", "url": "supp.png"}],  # supp has one figure
        ]

        with patch(
            "src.paper_loader.loaders.resolve_figure_url",
            return_value="http://supp.example.com/supp.png",
        ):
            with patch("src.paper_loader.loaders.generate_figure_id", return_value="supp1"):
                load_paper_from_markdown(
                    str(md_path),
                    str(tmp_path / "figs"),
                    base_url="http://main.example.com/",
                    supplementary_markdown_path=str(supp_path),
                    supplementary_base_url="http://supp.example.com/",
                    download_figures=True,
                )

        # Verify download was called with supplementary base URL
        mock_download.assert_called_once()
        args, _ = mock_download.call_args
        assert args[0] == "http://supp.example.com/supp.png"

    def test_supplementary_falls_back_to_base_url(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Supplementary uses base_url when supplementary_base_url not provided."""
        md_path = tmp_path / "main.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        supp_path = tmp_path / "supp.md"
        supp_path.write_text("Supp content", encoding="utf-8")

        mock_extract_figures.side_effect = [
            [],
            [{"alt": "Supp Fig", "url": "supp.png"}],
        ]

        with patch(
            "src.paper_loader.loaders.resolve_figure_url",
            return_value="http://main.example.com/supp.png",
        ):
            with patch("src.paper_loader.loaders.generate_figure_id", return_value="supp1"):
                load_paper_from_markdown(
                    str(md_path),
                    str(tmp_path / "figs"),
                    base_url="http://main.example.com/",
                    supplementary_markdown_path=str(supp_path),
                    download_figures=True,
                )

        mock_download.assert_called_once()
        args, _ = mock_download.call_args
        assert args[0] == "http://main.example.com/supp.png"

    def test_supplementary_figures_get_s_prefix(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Supplementary figures get 'S' prefix in their IDs."""
        md_path = tmp_path / "main.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        supp_path = tmp_path / "supp.md"
        supp_path.write_text("Supp content", encoding="utf-8")

        mock_extract_figures.side_effect = [
            [],
            [{"alt": "Supp Fig", "url": "supp.png"}],
        ]

        with patch("src.paper_loader.loaders.generate_figure_id", return_value="fig1"):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                supplementary_markdown_path=str(supp_path),
                download_figures=False,
            )

        assert len(paper["supplementary"]["supplementary_figures"]) == 1
        supp_fig = paper["supplementary"]["supplementary_figures"][0]
        assert supp_fig["id"].startswith("S")
        assert supp_fig["id"] == "Sfig1"

    def test_supplementary_figures_unique_from_main(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Supplementary figures avoid ID conflicts with main figures."""
        md_path = tmp_path / "main.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        supp_path = tmp_path / "supp.md"
        supp_path.write_text("Supp content", encoding="utf-8")

        mock_extract_figures.side_effect = [
            [{"alt": "Main Fig", "url": "main.png"}],
            [{"alt": "Supp Fig", "url": "supp.png"}],
        ]

        with patch("src.paper_loader.loaders.generate_figure_id", return_value="fig1"):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                supplementary_markdown_path=str(supp_path),
                download_figures=False,
            )

        assert len(paper["figures"]) == 1
        assert len(paper["supplementary"]["supplementary_figures"]) == 1
        
        main_id = paper["figures"][0]["id"]
        supp_id = paper["supplementary"]["supplementary_figures"][0]["id"]
        
        assert main_id == "fig1"
        assert supp_id == "Sfig1"
        assert main_id != supp_id

    def test_paper_length_warnings_logged(
        self, tmp_path, mock_extract_figures, caplog
    ):
        """Paper length warnings are logged."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        mock_extract_figures.return_value = []

        warnings = ["Paper is very long (>150K chars)"]
        with patch("src.paper_loader.loaders.check_paper_length", return_value=warnings):
            with caplog.at_level(logging.WARNING):
                paper = load_paper_from_markdown(
                    str(md_path),
                    str(tmp_path / "figs"),
                    download_figures=False,
                )

        assert any("Length warning" in record.message for record in caplog.records)
        assert paper["paper_id"] == "paper"  # Should still load successfully

    def test_validation_warnings_logged(
        self, tmp_path, mock_extract_figures, caplog
    ):
        """Validation warnings are logged but don't prevent loading."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        mock_extract_figures.return_value = []

        with patch("src.paper_loader.loaders.validate_paper_input", return_value=["Warning: paper_id has spaces"]):
            with caplog.at_level(logging.INFO):
                paper = load_paper_from_markdown(
                    str(md_path),
                    str(tmp_path / "figs"),
                    download_figures=False,
                )

        assert any("Validation warnings" in record.message for record in caplog.records)
        assert paper["paper_id"] == "paper"  # Should still load successfully

    def test_validation_errors_raise(
        self, tmp_path, mock_extract_figures
    ):
        """Validation errors raise ValidationError."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        mock_extract_figures.return_value = []

        from src.paper_loader.validation import ValidationError
        
        with patch("src.paper_loader.loaders.validate_paper_input", side_effect=ValidationError("Critical validation error")):
            with pytest.raises(ValidationError, match="Critical validation error"):
                load_paper_from_markdown(
                    str(md_path),
                    str(tmp_path / "figs"),
                    download_figures=False,
                )

    def test_figure_image_path_format(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Figure image_path uses correct format with figure ID and extension."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        mock_extract_figures.return_value = [{"alt": "Fig", "url": "figure.jpg"}]

        with patch("src.paper_loader.loaders.generate_figure_id", return_value="fig1"):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                download_figures=False,
            )

        assert len(paper["figures"]) == 1
        fig = paper["figures"][0]
        # Figures are now saved under {output_dir}/{paper_id}/run_{timestamp}/figures/
        run_output_dir = paper["run_output_dir"]
        expected_path = Path(run_output_dir) / "figures" / "fig1.jpg"
        assert fig["image_path"] == str(expected_path)
        assert fig["image_path"].endswith(".jpg")

    def test_download_figures_false_no_download(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """When download_figures=False, download is not called."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        mock_extract_figures.return_value = [{"alt": "Fig", "url": "fig.png"}]

        with patch("src.paper_loader.loaders.generate_figure_id", return_value="fig1"):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                download_figures=False,
            )

        mock_download.assert_not_called()
        assert len(paper["figures"]) == 1
        assert "download_error" not in paper["figures"][0]

    def test_download_figures_true_calls_download(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """When download_figures=True, download is called."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        mock_extract_figures.return_value = [{"alt": "Fig", "url": "fig.png"}]

        with patch("src.paper_loader.loaders.generate_figure_id", return_value="fig1"):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                download_figures=True,
            )

        mock_download.assert_called_once()
        assert len(paper["figures"]) == 1

    def test_supplementary_only_text_no_figures(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Supplementary section created with only text, no figures."""
        md_path = tmp_path / "main.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        supp_path = tmp_path / "supp.md"
        supp_path.write_text("Supp text only", encoding="utf-8")

        mock_extract_figures.side_effect = [[], []]  # No figures

        paper = load_paper_from_markdown(
            str(md_path),
            str(tmp_path / "figs"),
            supplementary_markdown_path=str(supp_path),
            download_figures=False,
        )

        assert "supplementary" in paper
        assert paper["supplementary"]["supplementary_text"] == "Supp text only"
        assert paper["supplementary"].get("supplementary_figures", []) == []

    def test_supplementary_only_figures_no_text(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Supplementary section created with only figures, empty text included."""
        md_path = tmp_path / "main.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        supp_path = tmp_path / "supp.md"
        supp_path.write_text("", encoding="utf-8")  # Empty text

        mock_extract_figures.side_effect = [
            [],
            [{"alt": "Supp Fig", "url": "supp.png"}],
        ]

        with patch("src.paper_loader.loaders.generate_figure_id", return_value="supp1"):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                supplementary_markdown_path=str(supp_path),
                download_figures=False,
            )

        # When supplementary_text is empty string (not None), it should be included
        assert "supplementary" in paper
        assert paper["supplementary"]["supplementary_text"] == ""
        assert len(paper["supplementary"]["supplementary_figures"]) == 1

    def test_all_required_paper_fields_present(
        self, tmp_path, mock_extract_figures
    ):
        """All required paper input fields are present in result."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(f"# Title\n{LONG_TEXT}", encoding="utf-8")
        mock_extract_figures.return_value = []

        paper = load_paper_from_markdown(
            str(md_path),
            str(tmp_path / "figs"),
            download_figures=False,
        )

        # Verify all required fields from schema
        required_fields = ["paper_id", "paper_title", "paper_text", "paper_domain", "figures"]
        for field in required_fields:
            assert field in paper, f"Missing required field: {field}"
        
        # Verify types
        assert isinstance(paper["paper_id"], str)
        assert isinstance(paper["paper_title"], str)
        assert isinstance(paper["paper_text"], str)
        assert isinstance(paper["paper_domain"], str)
        assert isinstance(paper["figures"], list)

    def test_figure_fields_complete(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """All required figure fields are present."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        mock_extract_figures.return_value = [{"alt": "Test Fig", "url": "test.png"}]

        with patch("src.paper_loader.loaders.generate_figure_id", return_value="test1"):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                download_figures=False,
            )

        assert len(paper["figures"]) == 1
        fig = paper["figures"][0]
        
        # Verify all required figure fields
        required_fields = ["id", "description", "image_path", "source_url"]
        for field in required_fields:
            assert field in fig, f"Missing required figure field: {field}"
        
        # Verify field types
        assert isinstance(fig["id"], str)
        assert isinstance(fig["description"], str)
        assert isinstance(fig["image_path"], str)
        assert isinstance(fig["source_url"], str)
        
        # Verify field values
        assert fig["id"] == "test1"
        assert fig["description"] == "Test Fig"
        assert fig["source_url"] == "test.png"
        # Figures are now saved under {output_dir}/{paper_id}/run_{timestamp}/figures/
        run_output_dir = paper["run_output_dir"]
        expected_path = Path(run_output_dir) / "figures" / "test1.png"
        assert fig["image_path"] == str(expected_path)

    def test_figure_output_directory_structure(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """
        Validates that figures are saved with the correct directory structure:
        {output_dir}/{paper_id}/run_{timestamp}/figures/
        
        This test ensures:
        1. The paper-specific directory is created
        2. The run-specific directory with timestamp is created
        3. The figures subdirectory is created inside it
        4. Figure image_path values reflect this structure
        5. Different paper_ids create separate directories
        6. run_output_dir is included in the returned paper
        """
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        mock_extract_figures.return_value = [{"alt": "Fig", "url": "fig.png"}]
        output_dir = tmp_path / "outputs"

        # Test with explicit paper_id
        with patch("src.paper_loader.loaders.generate_figure_id", return_value="fig1"):
            paper = load_paper_from_markdown(
                str(md_path),
                str(output_dir),
                paper_id="my_test_paper",
                download_figures=False,
            )

        # Verify directory structure was created
        paper_dir = output_dir / "my_test_paper"
        assert paper_dir.exists(), "Paper directory should be created"
        assert paper_dir.is_dir(), "Paper directory should be a directory"
        
        # Verify run directory was created
        run_dirs = list(paper_dir.glob("run_*"))
        assert len(run_dirs) == 1, "Exactly one run directory should be created"
        run_dir = run_dirs[0]
        assert run_dir.is_dir(), "Run directory should be a directory"
        assert run_dir.name.startswith("run_"), "Run directory should have run_ prefix"
        
        # Verify figures directory inside run directory
        figures_dir = run_dir / "figures"
        assert figures_dir.exists(), "Figures directory should be created"
        assert figures_dir.is_dir(), "Figures directory should be a directory"
        
        # Verify run_output_dir is returned
        assert "run_output_dir" in paper, "run_output_dir should be returned"
        assert paper["run_output_dir"] == str(run_dir)
        
        # Verify image_path reflects the correct structure
        fig = paper["figures"][0]
        expected_path = figures_dir / "fig1.png"
        assert fig["image_path"] == str(expected_path)
        assert "my_test_paper" in fig["image_path"]
        assert "run_" in fig["image_path"]
        assert "figures" in fig["image_path"]
        
        # Test that different paper_ids create separate directories
        mock_extract_figures.return_value = [{"alt": "Fig", "url": "fig.png"}]
        with patch("src.paper_loader.loaders.generate_figure_id", return_value="fig1"):
            paper2 = load_paper_from_markdown(
                str(md_path),
                str(output_dir),
                paper_id="another_paper",
                download_figures=False,
            )
        
        another_paper_dir = output_dir / "another_paper"
        assert another_paper_dir.exists(), "Second paper directory should be created"
        
        another_run_dirs = list(another_paper_dir.glob("run_*"))
        assert len(another_run_dirs) == 1, "Second paper should have one run directory"
        another_figures_dir = another_run_dirs[0] / "figures"
        assert another_figures_dir.exists(), "Second figures directory should be created"
        
        # Verify the two papers have different paths
        assert paper["figures"][0]["image_path"] != paper2["figures"][0]["image_path"]
        assert "my_test_paper" in paper["figures"][0]["image_path"]
        assert "another_paper" in paper2["figures"][0]["image_path"]
        
        # Verify run_output_dir is different for each paper
        assert paper["run_output_dir"] != paper2["run_output_dir"]


