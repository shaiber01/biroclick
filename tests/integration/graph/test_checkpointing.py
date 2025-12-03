"""Checkpointing integration tests for the LangGraph workflow."""

import json
from pathlib import Path
from unittest.mock import patch

from schemas.state import create_initial_state, save_checkpoint
from src.graph import generate_report_node_with_checkpoint


class TestCheckpointIntegration:
    """Tests that verify checkpoint behavior with actual file I/O."""

    def test_checkpoint_creates_file(self, tmp_path):
        """Test that checkpoints create actual files on disk."""
        state = create_initial_state(
            paper_id="checkpoint_test",
            paper_text="Test content for checkpoint",
        )

        checkpoint_path = save_checkpoint(
            state,
            "test_checkpoint",
            output_dir=str(tmp_path),
        )

        assert Path(checkpoint_path).exists()

    def test_checkpoint_contains_state_data(self, tmp_path):
        """Test that checkpoint file contains expected state data."""
        state = create_initial_state(
            paper_id="data_test",
            paper_text="Test content with data",
        )
        state["plan"] = {"stages": [{"stage_id": "stage_1"}]}

        checkpoint_path = save_checkpoint(
            state,
            "data_checkpoint",
            output_dir=str(tmp_path),
        )

        with open(checkpoint_path) as handle:
            saved_data = json.load(handle)

        assert saved_data["paper_id"] == "data_test"
        assert saved_data["paper_text"] == "Test content with data"
        assert "plan" in saved_data
        assert saved_data["plan"]["stages"][0]["stage_id"] == "stage_1"

    def test_checkpoint_creates_latest_link(self, tmp_path):
        """Test that checkpoint creates a 'latest' pointer file."""
        state = create_initial_state(
            paper_id="latest_test",
            paper_text="Test",
        )

        save_checkpoint(state, "my_checkpoint", output_dir=str(tmp_path))

        checkpoints_dir = tmp_path / "latest_test" / "checkpoints"
        latest_path = checkpoints_dir / "checkpoint_my_checkpoint_latest.json"

        assert latest_path.exists() or latest_path.is_symlink()


class TestReportNodeWrapper:
    """Tests for the generate_report_node_with_checkpoint wrapper."""

    @patch("src.graph._generate_report_node")
    @patch("src.graph.save_checkpoint")
    def test_report_wrapper_saves_checkpoint(self, mock_checkpoint, mock_report_node, test_state):
        """Test that the report wrapper saves a checkpoint after generating report."""
        mock_report_node.return_value = {"report_path": "/path/to/report.md"}

        result = generate_report_node_with_checkpoint(test_state)

        assert result == {"report_path": "/path/to/report.md"}

        mock_checkpoint.assert_called_once()
        saved_state, checkpoint_name = mock_checkpoint.call_args[0]
        assert checkpoint_name == "final_report"
        assert saved_state["report_path"] == "/path/to/report.md"


