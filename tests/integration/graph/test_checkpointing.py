"""Checkpointing integration tests for the LangGraph workflow."""

import json
import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from schemas.state import (
    create_initial_state,
    save_checkpoint,
    checkpoint_name_for_stage,
    load_checkpoint,
    list_checkpoints,
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

    def test_checkpoint_collision_handling(self, tmp_path):
        """Test that checkpoint handles filename collision (microsecond collision case)."""
        state = create_initial_state(
            paper_id="collision_test",
            paper_text="Test",
        )
        
        # Create checkpoint directory
        checkpoint_dir = tmp_path / "collision_test" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Manually create a checkpoint file with a specific timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        existing_file = checkpoint_dir / f"checkpoint_collision_test_collision_{timestamp}.json"
        existing_file.write_text('{"test": "data"}')
        
        # Now try to save with the same timestamp by mocking datetime
        with patch("schemas.state.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = timestamp
            
            checkpoint_path = save_checkpoint(
                state,
                "collision",
                output_dir=str(tmp_path),
            )
        
        # Should have created a file with a counter suffix
        assert Path(checkpoint_path).exists()
        # The path should be different from the existing file
        assert checkpoint_path != str(existing_file)
        
        # Both files should exist
        assert existing_file.exists()

    def test_checkpoint_unicode_in_paper_id(self, tmp_path):
        """Test checkpoint handles unicode characters in paper_id."""
        # Note: Most filesystems handle unicode, but let's verify
        state = create_initial_state(
            paper_id="test_αβγ_paper",  # Greek letters
            paper_text="Test",
        )
        
        try:
            checkpoint_path = save_checkpoint(
                state,
                "unicode_checkpoint",
                output_dir=str(tmp_path),
            )
            
            # Verify checkpoint was created
            assert Path(checkpoint_path).exists()
            
            # Verify it can be loaded back
            with open(checkpoint_path) as f:
                data = json.load(f)
            assert data["paper_id"] == "test_αβγ_paper"
        except OSError:
            # Some filesystems don't support unicode - that's OK
            pytest.skip("Filesystem doesn't support unicode in filenames")

    def test_checkpoint_with_spaces_in_checkpoint_name(self, tmp_path):
        """Test checkpoint handles spaces in checkpoint name."""
        state = create_initial_state(
            paper_id="space_test",
            paper_text="Test",
        )
        
        # Checkpoint name with spaces
        checkpoint_path = save_checkpoint(
            state,
            "my checkpoint name",
            output_dir=str(tmp_path),
        )
        
        assert Path(checkpoint_path).exists()
        assert "my checkpoint name" in checkpoint_path

    def test_checkpoint_with_dots_in_name(self, tmp_path):
        """Test checkpoint handles dots in paper_id and checkpoint name."""
        state = create_initial_state(
            paper_id="paper.v1.0",
            paper_text="Test",
        )
        
        checkpoint_path = save_checkpoint(
            state,
            "checkpoint.v2.0",
            output_dir=str(tmp_path),
        )
        
        assert Path(checkpoint_path).exists()
        
        # Verify directory structure
        expected_dir = tmp_path / "paper.v1.0" / "checkpoints"
        assert expected_dir.exists()

    def test_checkpoint_read_only_directory_error(self, tmp_path):
        """Test checkpoint fails gracefully with read-only output directory."""
        state = create_initial_state(
            paper_id="readonly_test",
            paper_text="Test",
        )
        
        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        
        try:
            # Make directory read-only
            readonly_dir.chmod(0o444)
            
            # Try to save checkpoint - should raise an error
            with pytest.raises((OSError, PermissionError)):
                save_checkpoint(
                    state,
                    "should_fail",
                    output_dir=str(readonly_dir),
                )
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)

    def test_checkpoint_very_large_state(self, tmp_path):
        """Test checkpoint handles very large state objects."""
        state = create_initial_state(
            paper_id="large_state_test",
            paper_text="Test paper " * 10000,  # ~100KB of text
        )
        
        # Add large nested structures
        state["large_list"] = list(range(10000))
        state["large_dict"] = {f"key_{i}": f"value_{i}" * 100 for i in range(1000)}
        state["deep_nesting"] = {"l1": {"l2": {"l3": {"l4": {"l5": {"data": "deep"}}}}}}
        
        checkpoint_path = save_checkpoint(
            state,
            "large_checkpoint",
            output_dir=str(tmp_path),
        )
        
        assert Path(checkpoint_path).exists()
        
        # Verify file size is substantial
        file_size = Path(checkpoint_path).stat().st_size
        assert file_size > 100000  # Should be > 100KB
        
        # Verify content is correct
        with open(checkpoint_path) as f:
            loaded = json.load(f)
        
        assert len(loaded["large_list"]) == 10000
        assert len(loaded["large_dict"]) == 1000
        assert loaded["deep_nesting"]["l1"]["l2"]["l3"]["l4"]["l5"]["data"] == "deep"

    def test_checkpoint_atomic_write(self, tmp_path):
        """Test that checkpoint write is complete (not partial/corrupt)."""
        state = create_initial_state(
            paper_id="atomic_test",
            paper_text="Test",
        )
        state["critical_data"] = {"key": "value"}
        
        checkpoint_path = save_checkpoint(
            state,
            "atomic_checkpoint",
            output_dir=str(tmp_path),
        )
        
        # Read the file and verify it's valid JSON
        with open(checkpoint_path) as f:
            content = f.read()
        
        # Should be valid JSON
        data = json.loads(content)
        assert data["critical_data"]["key"] == "value"
        
        # File should end properly (not truncated)
        assert content.strip().endswith("}")

    def test_checkpoint_preserves_float_precision(self, tmp_path):
        """Test that checkpoint preserves floating point precision."""
        state = create_initial_state(
            paper_id="float_test",
            paper_text="Test",
        )
        
        # Add floats with various precision requirements
        state["precise_floats"] = {
            "pi": 3.141592653589793,
            "small": 1e-15,
            "large": 1e15,
            "negative": -123.456789,
        }
        
        checkpoint_path = save_checkpoint(
            state,
            "float_checkpoint",
            output_dir=str(tmp_path),
        )
        
        with open(checkpoint_path) as f:
            loaded = json.load(f)
        
        # Verify float values are preserved
        assert abs(loaded["precise_floats"]["pi"] - 3.141592653589793) < 1e-10
        assert loaded["precise_floats"]["small"] == 1e-15
        assert loaded["precise_floats"]["large"] == 1e15
        assert loaded["precise_floats"]["negative"] == -123.456789

    def test_checkpoint_handles_datetime_serialization(self, tmp_path):
        """Test that checkpoint properly serializes datetime objects."""
        from datetime import datetime
        
        state = create_initial_state(
            paper_id="datetime_test",
            paper_text="Test",
        )
        
        now = datetime.now()
        state["timestamp"] = now
        state["date_string"] = now.isoformat()
        
        # save_checkpoint uses default=str to handle non-serializable types
        checkpoint_path = save_checkpoint(
            state,
            "datetime_checkpoint",
            output_dir=str(tmp_path),
        )
        
        assert Path(checkpoint_path).exists()
        
        with open(checkpoint_path) as f:
            loaded = json.load(f)
        
        # Datetime should be serialized as string
        assert isinstance(loaded["timestamp"], str)
        assert loaded["date_string"] == now.isoformat()


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
        
        # Verify all state fields are preserved in round trip
        for key in state:
            assert key in loaded, f"Key '{key}' missing after load"
            # For complex types, compare JSON serialized versions
            if isinstance(state[key], (dict, list)):
                assert json.dumps(state[key], sort_keys=True, default=str) == \
                       json.dumps(loaded[key], sort_keys=True, default=str), \
                       f"Key '{key}' has different value after load"

    def test_load_checkpoint_with_specific_name(self, tmp_path):
        """Test loading checkpoint using specific checkpoint name."""
        state = create_initial_state(
            paper_id="load_specific_test",
            paper_text="Test",
        )
        state["custom_field"] = "specific_value"

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
        assert loaded["custom_field"] == "specific_value"

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

    def test_load_checkpoint_directory_exists_but_no_matching_checkpoint(self, tmp_path):
        """Test loading when checkpoint directory exists but specific checkpoint doesn't."""
        state = create_initial_state(
            paper_id="partial_test",
            paper_text="Test",
        )
        
        # Save a checkpoint with one name
        save_checkpoint(state, "checkpoint_a", output_dir=str(tmp_path))
        
        # Try to load a different checkpoint name
        loaded = load_checkpoint(
            "partial_test",
            checkpoint_name="nonexistent_checkpoint",
            output_dir=str(tmp_path),
        )
        
        # Should return None because no checkpoint with that name exists
        assert loaded is None

    def test_load_checkpoint_multiple_checkpoints_returns_most_recent(self, tmp_path):
        """Test that 'latest' returns the most recently modified checkpoint."""
        state1 = create_initial_state(paper_id="multi_test", paper_text="First")
        state1["version"] = 1
        
        checkpoint1_path = save_checkpoint(state1, "first", output_dir=str(tmp_path))
        
        # Ensure distinct modification time
        time.sleep(0.05)
        
        state2 = create_initial_state(paper_id="multi_test", paper_text="Second")
        state2["version"] = 2
        
        checkpoint2_path = save_checkpoint(state2, "second", output_dir=str(tmp_path))
        
        # Load latest - should get the second checkpoint
        loaded = load_checkpoint(
            "multi_test",
            checkpoint_name="latest",
            output_dir=str(tmp_path),
        )
        
        assert loaded is not None
        assert loaded["paper_text"] == "Second"
        assert loaded["version"] == 2

    def test_load_checkpoint_with_timestamp_pattern_matching(self, tmp_path):
        """Test loading checkpoint by name finds the right one among multiple with same base name."""
        state1 = create_initial_state(paper_id="pattern_test", paper_text="First version")
        state1["data"] = "v1"
        
        save_checkpoint(state1, "my_checkpoint", output_dir=str(tmp_path))
        time.sleep(0.05)
        
        state2 = create_initial_state(paper_id="pattern_test", paper_text="Second version")
        state2["data"] = "v2"
        
        save_checkpoint(state2, "my_checkpoint", output_dir=str(tmp_path))
        
        # Load by specific name - should get the most recent one with that name
        loaded = load_checkpoint(
            "pattern_test",
            checkpoint_name="my_checkpoint",
            output_dir=str(tmp_path),
        )
        
        assert loaded is not None
        # Should be the second (most recent) checkpoint with that name
        assert loaded["data"] == "v2"

    def test_load_checkpoint_round_trip_preserves_all_data_types(self, tmp_path):
        """Test that save/load round trip preserves all JSON-serializable data types."""
        state = create_initial_state(
            paper_id="roundtrip_test",
            paper_text="Test",
        )
        
        # Add various data types
        state["string_field"] = "test string"
        state["int_field"] = 42
        state["float_field"] = 3.14159
        state["bool_true"] = True
        state["bool_false"] = False
        state["null_field"] = None
        state["list_field"] = [1, "two", 3.0, None, True]
        state["nested_dict"] = {
            "level1": {
                "level2": {
                    "value": "deep"
                }
            }
        }
        state["empty_list"] = []
        state["empty_dict"] = {}
        
        save_checkpoint(state, "roundtrip", output_dir=str(tmp_path))
        
        loaded = load_checkpoint(
            "roundtrip_test",
            checkpoint_name="latest",
            output_dir=str(tmp_path),
        )
        
        assert loaded is not None
        
        # Verify each type is preserved correctly
        assert loaded["string_field"] == "test string"
        assert loaded["int_field"] == 42
        assert isinstance(loaded["int_field"], int)
        assert loaded["float_field"] == 3.14159
        assert isinstance(loaded["float_field"], float)
        assert loaded["bool_true"] is True
        assert loaded["bool_false"] is False
        assert loaded["null_field"] is None
        assert loaded["list_field"] == [1, "two", 3.0, None, True]
        assert loaded["nested_dict"]["level1"]["level2"]["value"] == "deep"
        assert loaded["empty_list"] == []
        assert loaded["empty_dict"] == {}

    def test_load_checkpoint_handles_corrupt_json(self, tmp_path):
        """Test that load_checkpoint handles corrupt JSON file gracefully."""
        paper_id = "corrupt_test"
        checkpoint_dir = tmp_path / paper_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a corrupt JSON file
        corrupt_file = checkpoint_dir / "checkpoint_corrupt_test_my_checkpoint_20240101_120000_000000.json"
        corrupt_file.write_text("{invalid json: [}")
        
        # Attempting to load should raise an error (not silently fail)
        with pytest.raises(json.JSONDecodeError):
            load_checkpoint(
                paper_id,
                checkpoint_name="latest",
                output_dir=str(tmp_path),
            )

    def test_load_checkpoint_handles_empty_file(self, tmp_path):
        """Test that load_checkpoint handles empty JSON file."""
        paper_id = "empty_file_test"
        checkpoint_dir = tmp_path / paper_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create an empty file
        empty_file = checkpoint_dir / "checkpoint_empty_file_test_empty_20240101_120000_000000.json"
        empty_file.write_text("")
        
        # Attempting to load an empty file should raise JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            load_checkpoint(
                paper_id,
                checkpoint_name="latest",
                output_dir=str(tmp_path),
            )

    def test_load_checkpoint_symlink_to_deleted_file(self, tmp_path):
        """Test behavior when latest symlink points to a deleted file."""
        state = create_initial_state(paper_id="symlink_delete_test", paper_text="Test")
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(state, "to_delete", output_dir=str(tmp_path))
        
        # Delete the actual checkpoint file but keep the latest symlink
        checkpoint_dir = tmp_path / "symlink_delete_test" / "checkpoints"
        actual_file = Path(checkpoint_path)
        
        # Note: The latest file should either be a symlink or a copy
        latest_path = checkpoint_dir / "checkpoint_to_delete_latest.json"
        
        if latest_path.is_symlink():
            # Delete the target file
            actual_file.unlink()
            
            # Attempting to load should handle the broken symlink
            # The implementation finds most recent by glob, which won't include broken symlinks
            loaded = load_checkpoint(
                "symlink_delete_test",
                checkpoint_name="latest",
                output_dir=str(tmp_path),
            )
            # Should return None since the actual file is gone
            assert loaded is None
        else:
            # If it's a copy, this test doesn't apply - just verify it works
            loaded = load_checkpoint(
                "symlink_delete_test",
                checkpoint_name="to_delete",
                output_dir=str(tmp_path),
            )
            # If copy, the latest file still exists
            assert loaded is not None or actual_file.exists()

    def test_load_checkpoint_specific_name_via_latest_pointer(self, tmp_path):
        """Test loading specific checkpoint via its _latest.json pointer file."""
        state = create_initial_state(
            paper_id="pointer_test",
            paper_text="Pointer test content",
        )
        state["marker"] = "from_pointer"
        
        save_checkpoint(state, "my_named_checkpoint", output_dir=str(tmp_path))
        
        # Load by the specific name (should find via _latest.json pointer)
        loaded = load_checkpoint(
            "pointer_test",
            checkpoint_name="my_named_checkpoint",
            output_dir=str(tmp_path),
        )
        
        assert loaded is not None
        assert loaded["marker"] == "from_pointer"
        assert loaded["paper_id"] == "pointer_test"

    def test_load_checkpoint_unicode_content(self, tmp_path):
        """Test that checkpoints correctly handle unicode content."""
        state = create_initial_state(
            paper_id="unicode_test",
            paper_text="Test with unicode: \u03b1\u03b2\u03b3 (Greek), \u4e2d\u6587 (Chinese), \u0410\u0411\u0412 (Cyrillic)",
        )
        state["emoji"] = "Test with emoji: 🧪🔬📊"
        state["special"] = "Special chars: àéîöü ñ ß"
        
        save_checkpoint(state, "unicode_check", output_dir=str(tmp_path))
        
        loaded = load_checkpoint(
            "unicode_test",
            checkpoint_name="latest",
            output_dir=str(tmp_path),
        )
        
        assert loaded is not None
        assert "αβγ" in loaded["paper_text"]
        assert "中文" in loaded["paper_text"]
        assert loaded["emoji"] == "Test with emoji: 🧪🔬📊"
        assert loaded["special"] == "Special chars: àéîöü ñ ß"


class TestListCheckpoints:
    """Tests for list_checkpoints function."""

    def test_list_checkpoints_empty_directory(self, tmp_path):
        """Test listing checkpoints when no checkpoints exist."""
        result = list_checkpoints("nonexistent_paper", output_dir=str(tmp_path))
        
        assert result == []
        assert isinstance(result, list)

    def test_list_checkpoints_returns_all_checkpoints(self, tmp_path):
        """Test that list_checkpoints returns all saved checkpoints."""
        state = create_initial_state(paper_id="list_test", paper_text="Test")
        
        # Create multiple checkpoints
        save_checkpoint(state, "checkpoint_a", output_dir=str(tmp_path))
        time.sleep(0.01)
        save_checkpoint(state, "checkpoint_b", output_dir=str(tmp_path))
        time.sleep(0.01)
        save_checkpoint(state, "checkpoint_c", output_dir=str(tmp_path))
        
        result = list_checkpoints("list_test", output_dir=str(tmp_path))
        
        # Should have 3 checkpoints (not counting _latest files)
        assert len(result) == 3
        
        # Each should have required fields
        for cp in result:
            assert "name" in cp
            assert "timestamp" in cp
            assert "path" in cp
            assert "size_kb" in cp
            
            # Verify file exists
            assert Path(cp["path"]).exists()
            
            # Verify size is positive
            assert cp["size_kb"] > 0

    def test_list_checkpoints_excludes_latest_files(self, tmp_path):
        """Test that list_checkpoints excludes _latest.json pointer files."""
        state = create_initial_state(paper_id="list_excl_test", paper_text="Test")
        
        # Create a checkpoint (this also creates a _latest pointer)
        checkpoint_path = save_checkpoint(state, "my_checkpoint", output_dir=str(tmp_path))
        
        result = list_checkpoints("list_excl_test", output_dir=str(tmp_path))
        
        # Should have exactly 1 checkpoint (not 2)
        # The _latest.json pointer file should be excluded
        assert len(result) == 1
        
        # The one result should be the actual checkpoint, not the _latest pointer
        # Verify the result path matches the saved checkpoint path
        assert result[0]["path"] == checkpoint_path
        
        # The _latest pointer file should not be in the results
        # Check by verifying the filename doesn't end with _latest.json
        assert not result[0]["path"].endswith("_latest.json")
        
        # Verify the checkpoint directory has both files but only one is returned
        checkpoints_dir = tmp_path / "list_excl_test" / "checkpoints"
        all_files = list(checkpoints_dir.glob("checkpoint_*.json"))
        assert len(all_files) == 2  # Both actual checkpoint and _latest pointer
        
        # One should be _latest, one should be the actual checkpoint
        latest_files = [f for f in all_files if f.name.endswith("_latest.json")]
        actual_files = [f for f in all_files if not f.name.endswith("_latest.json")]
        assert len(latest_files) == 1
        assert len(actual_files) == 1

    def test_list_checkpoints_sorted_by_timestamp_descending(self, tmp_path):
        """Test that list_checkpoints returns results sorted by timestamp (most recent first)."""
        state = create_initial_state(paper_id="sort_test", paper_text="Test")
        
        # Create checkpoints with slight delays
        save_checkpoint(state, "first", output_dir=str(tmp_path))
        time.sleep(0.02)
        save_checkpoint(state, "second", output_dir=str(tmp_path))
        time.sleep(0.02)
        save_checkpoint(state, "third", output_dir=str(tmp_path))
        
        result = list_checkpoints("sort_test", output_dir=str(tmp_path))
        
        assert len(result) == 3
        
        # Should be sorted by timestamp descending (most recent first)
        timestamps = [cp["timestamp"] for cp in result]
        assert timestamps == sorted(timestamps, reverse=True)
        
        # The first entry should be the most recent (third checkpoint)
        assert "third" in result[0]["name"]

    def test_list_checkpoints_with_same_name_different_timestamps(self, tmp_path):
        """Test listing checkpoints when same name is used multiple times."""
        state = create_initial_state(paper_id="same_name_test", paper_text="Test")
        
        # Create multiple checkpoints with same name
        save_checkpoint(state, "repeated", output_dir=str(tmp_path))
        time.sleep(0.02)
        save_checkpoint(state, "repeated", output_dir=str(tmp_path))
        time.sleep(0.02)
        save_checkpoint(state, "repeated", output_dir=str(tmp_path))
        
        result = list_checkpoints("same_name_test", output_dir=str(tmp_path))
        
        # Should have 3 distinct checkpoints
        assert len(result) == 3
        
        # All should have the same name but different timestamps
        names = [cp["name"] for cp in result]
        timestamps = [cp["timestamp"] for cp in result]
        
        assert all(name == "repeated" for name in names)
        assert len(set(timestamps)) == 3  # All timestamps should be unique

    def test_list_checkpoints_includes_size(self, tmp_path):
        """Test that checkpoint size is accurately reported."""
        state1 = create_initial_state(paper_id="size_test", paper_text="Short")
        
        state2 = create_initial_state(paper_id="size_test", paper_text="Long " * 1000)
        state2["large_data"] = {"key" + str(i): "value" * 100 for i in range(100)}
        
        save_checkpoint(state1, "small", output_dir=str(tmp_path))
        save_checkpoint(state2, "large", output_dir=str(tmp_path))
        
        result = list_checkpoints("size_test", output_dir=str(tmp_path))
        
        # Find the checkpoints by name
        small_cp = next(cp for cp in result if "small" in cp["name"])
        large_cp = next(cp for cp in result if "large" in cp["name"])
        
        # Large checkpoint should have significantly larger size
        assert large_cp["size_kb"] > small_cp["size_kb"]
        
        # Sizes should be positive
        assert small_cp["size_kb"] > 0
        assert large_cp["size_kb"] > 0

    def test_list_checkpoints_path_is_absolute_or_resolvable(self, tmp_path):
        """Test that checkpoint paths are valid and can be used to load files."""
        state = create_initial_state(paper_id="path_test", paper_text="Test")
        
        save_checkpoint(state, "check_path", output_dir=str(tmp_path))
        
        result = list_checkpoints("path_test", output_dir=str(tmp_path))
        
        assert len(result) == 1
        path = result[0]["path"]
        
        # Path should exist and be readable
        assert Path(path).exists()
        
        # Should be able to load as JSON
        with open(path) as f:
            data = json.load(f)
        
        assert data["paper_id"] == "path_test"


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
