"""Checkpointing integration tests for the LangGraph workflow."""

import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from schemas.state import (
    create_initial_state,
    save_checkpoint,
    checkpoint_name_for_stage,
    load_checkpoint,
)
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

        # Verify file exists
        assert Path(checkpoint_path).exists()
        assert Path(checkpoint_path).is_file()
        
        # Verify path is absolute or correctly resolved
        assert os.path.isabs(checkpoint_path) or Path(checkpoint_path).resolve().exists()
        
        # Verify file is not empty
        assert Path(checkpoint_path).stat().st_size > 0
        
        # Verify file has correct extension
        assert checkpoint_path.endswith(".json")
        
        # Verify filename contains expected components
        filename = Path(checkpoint_path).name
        assert "checkpoint_" in filename
        assert "checkpoint_test" in filename
        assert "test_checkpoint" in filename

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

        # Verify all expected fields are present
        assert saved_data["paper_id"] == "data_test"
        assert saved_data["paper_text"] == "Test content with data"
        assert "plan" in saved_data
        assert saved_data["plan"]["stages"][0]["stage_id"] == "stage_1"
        
        # Verify state structure is preserved (not just top-level fields)
        assert isinstance(saved_data["plan"], dict)
        assert isinstance(saved_data["plan"]["stages"], list)
        assert len(saved_data["plan"]["stages"]) == 1
        
        # Verify JSON is valid and well-formed
        assert isinstance(saved_data, dict)
        
        # Verify original state fields are preserved
        assert saved_data.get("paper_domain") == state.get("paper_domain")
        assert saved_data.get("runtime_budget_minutes") == state.get("runtime_budget_minutes")

    def test_checkpoint_creates_latest_link(self, tmp_path):
        """Test that checkpoint creates a 'latest' pointer file."""
        state = create_initial_state(
            paper_id="latest_test",
            paper_text="Test",
        )

        save_checkpoint(state, "my_checkpoint", output_dir=str(tmp_path))

        checkpoints_dir = tmp_path / "latest_test" / "checkpoints"
        latest_path = checkpoints_dir / "checkpoint_my_checkpoint_latest.json"

        # Verify latest pointer exists (either as file or symlink)
        assert latest_path.exists() or latest_path.is_symlink()
        
        # Verify latest pointer points to valid checkpoint
        if latest_path.is_symlink():
            target = latest_path.readlink()
            assert (checkpoints_dir / target).exists()
        else:
            # If it's a copy, verify it's a valid JSON file
            assert latest_path.is_file()
            with open(latest_path) as f:
                json.load(f)  # Should not raise

    def test_checkpoint_creates_directory_structure(self, tmp_path):
        """Test that checkpoint creates nested directory structure."""
        state = create_initial_state(
            paper_id="nested_test",
            paper_text="Test",
        )

        checkpoint_path = save_checkpoint(
            state,
            "nested_checkpoint",
            output_dir=str(tmp_path),
        )

        # Verify directory structure was created
        expected_dir = tmp_path / "nested_test" / "checkpoints"
        assert expected_dir.exists()
        assert expected_dir.is_dir()
        
        # Verify checkpoint file is in correct location
        assert Path(checkpoint_path).parent == expected_dir

    def test_checkpoint_handles_missing_paper_id(self, tmp_path):
        """Test that checkpoint handles missing paper_id gracefully."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test",
        )
        # Remove paper_id to test fallback
        del state["paper_id"]

        checkpoint_path = save_checkpoint(
            state,
            "missing_id_checkpoint",
            output_dir=str(tmp_path),
        )

        # Should use "unknown" as fallback
        assert Path(checkpoint_path).exists()
        expected_dir = tmp_path / "unknown" / "checkpoints"
        assert expected_dir.exists()
        
        # Verify filename contains "unknown"
        filename = Path(checkpoint_path).name
        assert "unknown" in filename

    def test_checkpoint_handles_complex_nested_state(self, tmp_path):
        """Test that checkpoint correctly serializes complex nested state."""
        state = create_initial_state(
            paper_id="complex_test",
            paper_text="Test",
        )
        # Add complex nested structures
        state["plan"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "parameters": {"a": 1, "b": [1, 2, 3], "c": {"nested": "value"}},
                },
                {
                    "stage_id": "stage_1",
                    "metadata": {"tags": ["tag1", "tag2"], "count": 42},
                },
            ],
            "metadata": {"version": "1.0", "created": "2024-01-01"},
        }
        state["assumptions"] = [
            {"parameter": "wavelength", "value": 500, "source": "figure"},
            {"parameter": "material", "value": "gold", "source": "text"},
        ]

        checkpoint_path = save_checkpoint(
            state,
            "complex_checkpoint",
            output_dir=str(tmp_path),
        )

        with open(checkpoint_path) as handle:
            saved_data = json.load(handle)

        # Verify complex nested structures are preserved
        assert len(saved_data["plan"]["stages"]) == 2
        assert saved_data["plan"]["stages"][0]["parameters"]["a"] == 1
        assert saved_data["plan"]["stages"][0]["parameters"]["b"] == [1, 2, 3]
        assert saved_data["plan"]["stages"][0]["parameters"]["c"]["nested"] == "value"
        assert saved_data["plan"]["stages"][1]["metadata"]["tags"] == ["tag1", "tag2"]
        assert saved_data["plan"]["stages"][1]["metadata"]["count"] == 42
        assert saved_data["plan"]["metadata"]["version"] == "1.0"
        assert len(saved_data["assumptions"]) == 2
        assert saved_data["assumptions"][0]["parameter"] == "wavelength"
        assert saved_data["assumptions"][1]["source"] == "text"

    def test_checkpoint_handles_none_values(self, tmp_path):
        """Test that checkpoint handles None values correctly."""
        state = create_initial_state(
            paper_id="none_test",
            paper_text="Test",
        )
        state["plan"] = None
        state["current_stage_id"] = None
        state["nested"] = {"key": None, "other": "value"}

        checkpoint_path = save_checkpoint(
            state,
            "none_checkpoint",
            output_dir=str(tmp_path),
        )

        with open(checkpoint_path) as handle:
            saved_data = json.load(handle)

        # Verify None values are preserved (JSON null)
        assert saved_data["plan"] is None
        assert saved_data["current_stage_id"] is None
        assert saved_data["nested"]["key"] is None
        assert saved_data["nested"]["other"] == "value"

    def test_checkpoint_handles_empty_state(self, tmp_path):
        """Test that checkpoint handles minimal state."""
        state = create_initial_state(
            paper_id="empty_test",
            paper_text="",
        )

        checkpoint_path = save_checkpoint(
            state,
            "empty_checkpoint",
            output_dir=str(tmp_path),
        )

        assert Path(checkpoint_path).exists()
        
        with open(checkpoint_path) as handle:
            saved_data = json.load(handle)

        # Verify basic fields exist even with empty text
        assert saved_data["paper_id"] == "empty_test"
        assert saved_data["paper_text"] == ""

    def test_checkpoint_handles_special_characters_in_paper_id(self, tmp_path):
        """Test that checkpoint handles special characters in paper_id."""
        state = create_initial_state(
            paper_id="test_paper-123",
            paper_text="Test",
        )

        checkpoint_path = save_checkpoint(
            state,
            "special_checkpoint",
            output_dir=str(tmp_path),
        )

        assert Path(checkpoint_path).exists()
        
        # Verify directory was created with special characters
        expected_dir = tmp_path / "test_paper-123" / "checkpoints"
        assert expected_dir.exists()

    def test_checkpoint_handles_long_checkpoint_names(self, tmp_path):
        """Test that checkpoint handles long checkpoint names."""
        state = create_initial_state(
            paper_id="long_test",
            paper_text="Test",
        )

        long_name = "a" * 200  # Very long name
        checkpoint_path = save_checkpoint(
            state,
            long_name,
            output_dir=str(tmp_path),
        )

        assert Path(checkpoint_path).exists()
        
        # Verify filename contains the long name (may be truncated by filesystem)
        filename = Path(checkpoint_path).name
        assert long_name[:50] in filename or "checkpoint_" in filename

    def test_checkpoint_returns_correct_path(self, tmp_path):
        """Test that checkpoint returns the correct file path."""
        state = create_initial_state(
            paper_id="path_test",
            paper_text="Test",
        )

        checkpoint_path = save_checkpoint(
            state,
            "path_checkpoint",
            output_dir=str(tmp_path),
        )

        # Verify returned path matches actual file
        assert Path(checkpoint_path).exists()
        assert Path(checkpoint_path).is_file()
        
        # Verify path format
        assert checkpoint_path.endswith(".json")
        assert "checkpoint_" in Path(checkpoint_path).name

    def test_checkpoint_overwrites_latest_link(self, tmp_path):
        """Test that checkpoint overwrites existing latest link."""
        state = create_initial_state(
            paper_id="overwrite_test",
            paper_text="Test",
        )

        # Create first checkpoint
        first_path = save_checkpoint(
            state,
            "overwrite_checkpoint",
            output_dir=str(tmp_path),
        )

        # Create second checkpoint with same name
        second_path = save_checkpoint(
            state,
            "overwrite_checkpoint",
            output_dir=str(tmp_path),
        )

        # Verify both checkpoints exist
        assert Path(first_path).exists()
        assert Path(second_path).exists()
        assert first_path != second_path

        # Verify latest link points to the second (most recent) checkpoint
        checkpoints_dir = tmp_path / "overwrite_test" / "checkpoints"
        latest_path = checkpoints_dir / "checkpoint_overwrite_checkpoint_latest.json"
        assert latest_path.exists() or latest_path.is_symlink()
        
        # Verify latest points to second checkpoint
        if latest_path.is_symlink():
            assert str(latest_path.readlink()) == Path(second_path).name
        else:
            # If copy, verify content matches second checkpoint
            with open(latest_path) as f1, open(second_path) as f2:
                assert json.load(f1) == json.load(f2)

    def test_checkpoint_handles_symlink_failure_gracefully(self, tmp_path):
        """Test that checkpoint falls back to copy when symlink fails."""
        state = create_initial_state(
            paper_id="symlink_test",
            paper_text="Test",
        )

        # Mock symlink to fail
        with patch("pathlib.Path.symlink_to", side_effect=OSError("Symlink failed")):
            checkpoint_path = save_checkpoint(
                state,
                "symlink_checkpoint",
                output_dir=str(tmp_path),
            )

        # Verify checkpoint was still created
        assert Path(checkpoint_path).exists()
        
        # Verify latest pointer exists (should be a copy, not symlink)
        checkpoints_dir = tmp_path / "symlink_test" / "checkpoints"
        latest_path = checkpoints_dir / "checkpoint_symlink_checkpoint_latest.json"
        assert latest_path.exists()
        
        # Verify it's a file (copy), not a symlink
        assert latest_path.is_file()
        assert not latest_path.is_symlink()
        
        # Verify content matches
        with open(latest_path) as f1, open(checkpoint_path) as f2:
            assert json.load(f1) == json.load(f2)

    def test_checkpoint_handles_notimplemented_error(self, tmp_path):
        """Test that checkpoint handles NotImplementedError from symlink."""
        state = create_initial_state(
            paper_id="notimpl_test",
            paper_text="Test",
        )

        # Mock symlink to raise NotImplementedError (Windows without Developer Mode)
        with patch("pathlib.Path.symlink_to", side_effect=NotImplementedError("Not supported")):
            checkpoint_path = save_checkpoint(
                state,
                "notimpl_checkpoint",
                output_dir=str(tmp_path),
            )

        # Verify checkpoint was still created
        assert Path(checkpoint_path).exists()
        
        # Verify latest pointer exists as copy
        checkpoints_dir = tmp_path / "notimpl_test" / "checkpoints"
        latest_path = checkpoints_dir / "checkpoint_notimpl_checkpoint_latest.json"
        assert latest_path.exists()
        assert latest_path.is_file()

    def test_checkpoint_handles_existing_latest_file(self, tmp_path):
        """Test that checkpoint removes existing latest file before creating new one."""
        state = create_initial_state(
            paper_id="existing_test",
            paper_text="Test",
        )

        checkpoints_dir = tmp_path / "existing_test" / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a file that would conflict with latest pointer
        latest_path = checkpoints_dir / "checkpoint_test_latest.json"
        latest_path.write_text('{"old": "data"}')

        # Save checkpoint
        checkpoint_path = save_checkpoint(
            state,
            "test",
            output_dir=str(tmp_path),
        )

        # Verify old file was replaced
        assert latest_path.exists()
        with open(latest_path) as f:
            data = json.load(f)
            assert data.get("paper_id") == "existing_test"
            assert "old" not in data

    def test_checkpoint_handles_existing_symlink(self, tmp_path):
        """Test that checkpoint removes existing symlink before creating new one."""
        state = create_initial_state(
            paper_id="symlink_existing_test",
            paper_text="Test",
        )

        checkpoints_dir = tmp_path / "symlink_existing_test" / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Create an old checkpoint file
        old_checkpoint = checkpoints_dir / "checkpoint_old.json"
        old_checkpoint.write_text('{"old": "data"}')
        
        # Create a symlink pointing to old checkpoint
        latest_path = checkpoints_dir / "checkpoint_test_latest.json"
        try:
            latest_path.symlink_to("checkpoint_old.json")
        except (OSError, NotImplementedError):
            # If symlinks aren't supported, skip this test
            pytest.skip("Symlinks not supported on this system")

        # Save checkpoint
        checkpoint_path = save_checkpoint(
            state,
            "test",
            output_dir=str(tmp_path),
        )

        # Verify symlink was replaced (either with new symlink or copy)
        assert latest_path.exists()
        # Should point to new checkpoint, not old one
        if latest_path.is_symlink():
            assert latest_path.readlink() != "checkpoint_old.json"

    def test_checkpoint_preserves_all_state_fields(self, tmp_path):
        """Test that checkpoint preserves all state fields, not just a subset."""
        state = create_initial_state(
            paper_id="preserve_test",
            paper_text="Test",
        )
        
        # Add many fields
        state["plan"] = {"stages": []}
        state["current_stage_id"] = "stage_0"
        state["assumptions"] = []
        state["metrics"] = {"tokens_used": 1000}
        state["validated_materials"] = ["gold", "silver"]
        state["user_responses"] = {"question1": "answer1"}
        state["supervisor_verdict"] = "ok_continue"

        checkpoint_path = save_checkpoint(
            state,
            "preserve_checkpoint",
            output_dir=str(tmp_path),
        )

        with open(checkpoint_path) as handle:
            saved_data = json.load(handle)

        # Verify all fields are present
        assert saved_data["plan"] == state["plan"]
        assert saved_data["current_stage_id"] == state["current_stage_id"]
        assert saved_data["assumptions"] == state["assumptions"]
        assert saved_data["metrics"] == state["metrics"]
        assert saved_data["validated_materials"] == state["validated_materials"]
        assert saved_data["user_responses"] == state["user_responses"]
        assert saved_data["supervisor_verdict"] == state["supervisor_verdict"]
        
        # Verify no fields were lost
        original_keys = set(state.keys())
        saved_keys = set(saved_data.keys())
        assert original_keys == saved_keys, f"Missing keys: {original_keys - saved_keys}"

    def test_checkpoint_handles_json_serialization_edge_cases(self, tmp_path):
        """Test that checkpoint handles JSON serialization edge cases."""
        state = create_initial_state(
            paper_id="json_test",
            paper_text="Test",
        )
        
        # Add types that need special JSON handling
        from datetime import datetime
        state["timestamp"] = datetime.now()
        state["numbers"] = {
            "int": 42,
            "float": 3.14159,
            "negative": -10,
            "zero": 0,
            "large": 10**10,
        }
        state["booleans"] = {"true": True, "false": False}
        state["empty"] = {"list": [], "dict": {}}

        checkpoint_path = save_checkpoint(
            state,
            "json_checkpoint",
            output_dir=str(tmp_path),
        )

        # Should not raise during serialization
        assert Path(checkpoint_path).exists()
        
        with open(checkpoint_path) as handle:
            saved_data = json.load(handle)

        # Verify numbers are preserved correctly
        assert saved_data["numbers"]["int"] == 42
        assert saved_data["numbers"]["float"] == 3.14159
        assert saved_data["numbers"]["negative"] == -10
        assert saved_data["numbers"]["zero"] == 0
        assert saved_data["numbers"]["large"] == 10**10
        
        # Verify booleans are preserved
        assert saved_data["booleans"]["true"] is True
        assert saved_data["booleans"]["false"] is False
        
        # Verify empty structures are preserved
        assert saved_data["empty"]["list"] == []
        assert saved_data["empty"]["dict"] == {}
        
        # Timestamp should be serialized (using default=str)
        assert "timestamp" in saved_data

    def test_checkpoint_creates_unique_timestamps(self, tmp_path):
        """Test that checkpoints have unique timestamps."""
        state = create_initial_state(
            paper_id="timestamp_test",
            paper_text="Test",
        )

        checkpoint_path1 = save_checkpoint(
            state,
            "timestamp_checkpoint",
            output_dir=str(tmp_path),
        )
        
        # Small delay to ensure different timestamp
        import time
        time.sleep(0.01)
        
        checkpoint_path2 = save_checkpoint(
            state,
            "timestamp_checkpoint",
            output_dir=str(tmp_path),
        )

        # Verify filenames are different
        assert checkpoint_path1 != checkpoint_path2
        
        # Verify both files exist
        assert Path(checkpoint_path1).exists()
        assert Path(checkpoint_path2).exists()
        
        # Verify timestamps in filenames are different
        filename1 = Path(checkpoint_path1).name
        filename2 = Path(checkpoint_path2).name
        assert filename1 != filename2


class TestCheckpointNameGeneration:
    """Tests for checkpoint_name_for_stage helper function."""

    def test_checkpoint_name_for_stage_with_valid_stage_id(self):
        """Test checkpoint name generation with valid stage_id."""
        state = {"current_stage_id": "stage0_material_validation"}
        
        name = checkpoint_name_for_stage(state, "complete")
        assert name == "stage0_complete"

    def test_checkpoint_name_for_stage_with_different_stage_numbers(self):
        """Test checkpoint name generation with different stage numbers."""
        state1 = {"current_stage_id": "stage1_single_structure"}
        state2 = {"current_stage_id": "stage2_parameter_sweep"}
        state3 = {"current_stage_id": "stage10_custom_stage"}
        
        assert checkpoint_name_for_stage(state1, "complete") == "stage1_complete"
        assert checkpoint_name_for_stage(state2, "user_confirm") == "stage2_user_confirm"
        assert checkpoint_name_for_stage(state3, "complete") == "stage10_complete"

    def test_checkpoint_name_for_stage_with_missing_stage_id(self):
        """Test checkpoint name generation with missing stage_id."""
        state = {}
        
        name = checkpoint_name_for_stage(state, "complete")
        assert name == "unknown_complete"

    def test_checkpoint_name_for_stage_with_none_stage_id(self):
        """Test checkpoint name generation with None stage_id."""
        state = {"current_stage_id": None}
        
        name = checkpoint_name_for_stage(state, "complete")
        assert name == "unknown_complete"

    def test_checkpoint_name_for_stage_with_invalid_format(self):
        """Test checkpoint name generation with invalid stage_id format."""
        state = {"current_stage_id": "invalid_format"}
        
        name = checkpoint_name_for_stage(state, "complete")
        assert name == "invalid_format_complete"

    def test_checkpoint_name_for_stage_with_different_events(self):
        """Test checkpoint name generation with different event types."""
        state = {"current_stage_id": "stage5_custom"}
        
        assert checkpoint_name_for_stage(state, "complete") == "stage5_complete"
        assert checkpoint_name_for_stage(state, "user_confirm") == "stage5_user_confirm"
        assert checkpoint_name_for_stage(state, "error") == "stage5_error"


class TestLoadCheckpoint:
    """Tests for load_checkpoint function."""

    def test_load_checkpoint_with_latest(self, tmp_path):
        """Test loading checkpoint using 'latest'."""
        state = create_initial_state(
            paper_id="load_test",
            paper_text="Test content",
        )
        state["plan"] = {"stages": [{"stage_id": "stage_1"}]}

        # Save checkpoint
        checkpoint_path = save_checkpoint(
            state,
            "load_checkpoint",
            output_dir=str(tmp_path),
        )

        # Load using 'latest'
        loaded = load_checkpoint(
            "load_test",
            checkpoint_name="latest",
            output_dir=str(tmp_path),
        )

        assert loaded is not None
        assert loaded["paper_id"] == "load_test"
        assert loaded["paper_text"] == "Test content"
        assert loaded["plan"]["stages"][0]["stage_id"] == "stage_1"

    def test_load_checkpoint_with_specific_name(self, tmp_path):
        """Test loading checkpoint using specific checkpoint name."""
        state = create_initial_state(
            paper_id="load_specific_test",
            paper_text="Test",
        )

        save_checkpoint(
            state,
            "specific_checkpoint",
            output_dir=str(tmp_path),
        )

        # Load using specific name
        loaded = load_checkpoint(
            "load_specific_test",
            checkpoint_name="specific_checkpoint",
            output_dir=str(tmp_path),
        )

        assert loaded is not None
        assert loaded["paper_id"] == "load_specific_test"

    def test_load_checkpoint_returns_none_when_not_found(self, tmp_path):
        """Test that load_checkpoint returns None when checkpoint doesn't exist."""
        loaded = load_checkpoint(
            "nonexistent",
            checkpoint_name="latest",
            output_dir=str(tmp_path),
        )

        assert loaded is None

    def test_load_checkpoint_handles_missing_directory(self, tmp_path):
        """Test that load_checkpoint handles missing checkpoint directory."""
        loaded = load_checkpoint(
            "missing_dir_test",
            checkpoint_name="any_checkpoint",
            output_dir=str(tmp_path),
        )

        assert loaded is None


class TestReportNodeWrapper:
    """Tests for the generate_report_node_with_checkpoint wrapper."""

    @patch("src.graph._generate_report_node")
    @patch("src.graph.save_checkpoint")
    def test_report_wrapper_saves_checkpoint(self, mock_checkpoint, mock_report_node, test_state):
        """Test that the report wrapper saves a checkpoint after generating report."""
        mock_report_node.return_value = {"report_path": "/path/to/report.md"}

        result = generate_report_node_with_checkpoint(test_state)

        # Verify return value
        assert result == {"report_path": "/path/to/report.md"}
        
        # Verify report node was called with correct state
        mock_report_node.assert_called_once_with(test_state)
        
        # Verify checkpoint was saved
        mock_checkpoint.assert_called_once()
        
        # Verify checkpoint was called with correct arguments
        call_args = mock_checkpoint.call_args
        saved_state, checkpoint_name = call_args[0]
        assert checkpoint_name == "final_report"
        
        # Verify saved state includes report_path
        assert saved_state["report_path"] == "/path/to/report.md"
        
        # Verify saved state includes original state fields
        assert saved_state["paper_id"] == test_state["paper_id"]
        assert saved_state["paper_text"] == test_state["paper_text"]

    @patch("src.graph._generate_report_node")
    @patch("src.graph.save_checkpoint")
    def test_report_wrapper_merges_state_correctly(self, mock_checkpoint, mock_report_node, test_state):
        """Test that wrapper correctly merges report result into state."""
        test_state["plan"] = {"stages": []}
        test_state["current_stage_id"] = "stage_0"
        
        mock_report_node.return_value = {
            "report_path": "/path/to/report.md",
            "report_generated_at": "2024-01-01",
        }

        result = generate_report_node_with_checkpoint(test_state)

        # Verify checkpoint was called with merged state
        mock_checkpoint.assert_called_once()
        saved_state, _ = mock_checkpoint.call_args[0]
        
        # Verify original state fields are preserved
        assert saved_state["paper_id"] == test_state["paper_id"]
        assert saved_state["plan"] == test_state["plan"]
        assert saved_state["current_stage_id"] == test_state["current_stage_id"]
        
        # Verify report fields are added
        assert saved_state["report_path"] == "/path/to/report.md"
        assert saved_state["report_generated_at"] == "2024-01-01"

    @patch("src.graph._generate_report_node")
    @patch("src.graph.save_checkpoint")
    def test_report_wrapper_handles_empty_result(self, mock_checkpoint, mock_report_node, test_state):
        """Test that wrapper handles empty result from report node."""
        mock_report_node.return_value = {}

        result = generate_report_node_with_checkpoint(test_state)

        assert result == {}
        
        # Verify checkpoint was still saved
        mock_checkpoint.assert_called_once()
        saved_state, _ = mock_checkpoint.call_args[0]
        
        # Verify state is preserved even with empty result
        assert saved_state["paper_id"] == test_state["paper_id"]

    @patch("src.graph._generate_report_node")
    @patch("src.graph.save_checkpoint")
    def test_report_wrapper_preserves_all_state_fields(self, mock_checkpoint, mock_report_node, test_state):
        """Test that wrapper preserves all state fields when merging."""
        test_state["plan"] = {"stages": []}
        test_state["assumptions"] = [{"param": "wavelength", "value": 500}]
        test_state["metrics"] = {"tokens_used": 1000}
        
        mock_report_node.return_value = {"report_path": "/path/to/report.md"}

        generate_report_node_with_checkpoint(test_state)

        # Verify all fields are preserved in saved state
        mock_checkpoint.assert_called_once()
        saved_state, _ = mock_checkpoint.call_args[0]
        
        assert saved_state["plan"] == test_state["plan"]
        assert saved_state["assumptions"] == test_state["assumptions"]
        assert saved_state["metrics"] == test_state["metrics"]
        assert saved_state["report_path"] == "/path/to/report.md"

    @patch("src.graph._generate_report_node")
    @patch("src.graph.save_checkpoint")
    def test_report_wrapper_handles_report_node_error(self, mock_checkpoint, mock_report_node, test_state):
        """Test that wrapper propagates errors from report node."""
        mock_report_node.side_effect = ValueError("Report generation failed")

        with pytest.raises(ValueError, match="Report generation failed"):
            generate_report_node_with_checkpoint(test_state)

        # Verify checkpoint was NOT saved when report node fails
        mock_checkpoint.assert_not_called()

    @patch("src.graph._generate_report_node")
    @patch("src.graph.save_checkpoint")
    def test_report_wrapper_handles_checkpoint_error(self, mock_checkpoint, mock_report_node, test_state):
        """Test that wrapper propagates errors from checkpoint save."""
        mock_report_node.return_value = {"report_path": "/path/to/report.md"}
        mock_checkpoint.side_effect = IOError("Failed to save checkpoint")

        # Should propagate checkpoint error
        with pytest.raises(IOError, match="Failed to save checkpoint"):
            generate_report_node_with_checkpoint(test_state)

        # Verify report node was called successfully
        mock_report_node.assert_called_once()

    @patch("src.graph._generate_report_node")
    @patch("src.graph.save_checkpoint")
    def test_report_wrapper_overwrites_existing_keys(self, mock_checkpoint, mock_report_node, test_state):
        """Test that wrapper overwrites existing keys when merging."""
        test_state["report_path"] = "/old/path.md"
        
        mock_report_node.return_value = {"report_path": "/new/path.md"}

        result = generate_report_node_with_checkpoint(test_state)

        assert result["report_path"] == "/new/path.md"
        
        # Verify saved state has new value
        mock_checkpoint.assert_called_once()
        saved_state, _ = mock_checkpoint.call_args[0]
        assert saved_state["report_path"] == "/new/path.md"
        assert saved_state["report_path"] != "/old/path.md"
