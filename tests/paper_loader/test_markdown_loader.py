"""Tests for loading paper inputs from markdown sources."""

import logging
from unittest.mock import patch

import pytest

from src.paper_loader import load_paper_from_markdown
from src.paper_loader.downloader import FigureDownloadError


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
        """Tests basic markdown loading."""
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

        assert paper["paper_title"] == "Title"
        assert paper["paper_text"] == text
        assert paper["figures"] == []
        assert paper["paper_id"] == "test"

    def test_loads_markdown_extracts_figures(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Extracts figure references from markdown."""
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
        assert fig["description"] == "Fig 1"
        assert fig["id"] == "fig1"
        mock_download.assert_called_once()

    def test_generates_unique_ids_for_duplicates(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Generates unique IDs for duplicate figures."""
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
        id1 = paper["figures"][0]["id"]
        id2 = paper["figures"][1]["id"]
        assert id1 == "fig1"
        assert id2 == "fig1_1"

    def test_with_base_url(self, tmp_path, mock_extract_figures, mock_download):
        """Resolves URLs with base_url."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")

        mock_extract_figures.return_value = [{"alt": "Fig", "url": "relative/fig.png"}]

        with patch(
            "src.paper_loader.loaders.resolve_figure_url",
            return_value="http://example.com/relative/fig.png",
        ):
            load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "figs"),
                base_url="http://example.com/",
                download_figures=True,
            )

        args, _ = mock_download.call_args
        url = args[0]
        assert url == "http://example.com/relative/fig.png"

    def test_supplementary_markdown(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Loads supplementary markdown and figures."""
        md_path = tmp_path / "main.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        supp_path = tmp_path / "supp.md"
        supp_path.write_text("Supp content " + LONG_TEXT, encoding="utf-8")

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

        assert paper["supplementary"]["supplementary_text"].startswith("Supp content")
        assert len(paper["figures"]) == 1
        assert len(paper["supplementary"]["supplementary_figures"]) == 1

        supp_fig = paper["supplementary"]["supplementary_figures"][0]
        assert supp_fig["id"] == "Sfigure_supp"

    def test_download_error_handling(
        self, tmp_path, mock_extract_figures, mock_download
    ):
        """Handles download errors gracefully."""
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
        assert "download_error" in paper["figures"][0]
        assert "404 Not Found" in paper["figures"][0]["download_error"]

    def test_file_not_found_raises(self, tmp_path):
        """Raises FileNotFoundError for non-existent markdown."""
        with pytest.raises(FileNotFoundError, match="Markdown file not found"):
            load_paper_from_markdown(
                markdown_path="/nonexistent/paper.md", output_dir=str(tmp_path)
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

    def test_output_dir_creation(self, tmp_path, mock_extract_figures):
        """Creates output directory if it doesn't exist."""
        md_path = tmp_path / "paper.md"
        md_path.write_text(LONG_TEXT, encoding="utf-8")
        mock_extract_figures.return_value = []

        out_dir = tmp_path / "nested" / "dirs"
        assert not out_dir.exists()

        load_paper_from_markdown(
            str(md_path),
            str(out_dir),
            download_figures=False,
        )

        assert out_dir.exists()


