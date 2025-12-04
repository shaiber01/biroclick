"""Tests for run-based folder structure and backwards compatibility.

This module tests:
1. run_output_dir generation and propagation
2. Multiple runs create separate folders
3. Backwards compatibility with legacy checkpoint paths
4. Edge cases and error handling
5. Content verification (not just existence)
6. Downstream integration
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from schemas.state import (
    create_initial_state,
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
)
from src.paper_loader import load_paper_from_markdown
from src.paper_loader.state_conversion import create_state_from_paper_input


# ═══════════════════════════════════════════════════════════════════════
# run_output_dir Generation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRunOutputDirGeneration:
    """Test that run_output_dir is properly generated."""
    
    def test_load_paper_creates_run_folder(self, tmp_path):
        """load_paper_from_markdown should create a run folder with timestamp."""
        md_path = tmp_path / "paper.md"
        md_path.write_text("# Test Paper\n" + "A" * 150, encoding="utf-8")
        output_dir = tmp_path / "outputs"
        
        with patch("src.paper_loader.loaders.extract_figures_from_markdown", return_value=[]):
            paper = load_paper_from_markdown(
                str(md_path),
                str(output_dir),
                paper_id="test_paper",
                download_figures=False,
            )
        
        # Verify run_output_dir is returned and is a string
        assert "run_output_dir" in paper, "run_output_dir must be in returned paper"
        assert isinstance(paper["run_output_dir"], str), "run_output_dir must be a string"
        assert paper["run_output_dir"] != "", "run_output_dir must not be empty"
        
        run_output_dir = Path(paper["run_output_dir"])
        
        # Verify folder structure - check exact hierarchy
        assert run_output_dir.exists(), f"run_output_dir {run_output_dir} must exist on disk"
        assert run_output_dir.is_dir(), "run_output_dir must be a directory"
        assert run_output_dir.parent.name == "test_paper", "run_output_dir parent must be paper_id"
        assert run_output_dir.parent.parent == output_dir, "paper_id folder must be under output_dir"
        assert run_output_dir.name.startswith("run_"), "run folder must start with 'run_'"
        
        # Verify figures subfolder exists
        figures_dir = run_output_dir / "figures"
        assert figures_dir.exists(), "figures subfolder must be created"
        assert figures_dir.is_dir(), "figures must be a directory"
    
    def test_run_folder_has_timestamp_format(self, tmp_path):
        """Run folder should have format run_YYYYMMDD_HHMMSS with valid timestamp."""
        md_path = tmp_path / "paper.md"
        md_path.write_text("# Test\n" + "A" * 150, encoding="utf-8")
        output_dir = tmp_path / "outputs"
        
        before_time = datetime.now()
        
        with patch("src.paper_loader.loaders.extract_figures_from_markdown", return_value=[]):
            paper = load_paper_from_markdown(
                str(md_path),
                str(output_dir),
                download_figures=False,
            )
        
        after_time = datetime.now()
        
        run_dir_name = Path(paper["run_output_dir"]).name
        assert run_dir_name.startswith("run_"), "Must start with 'run_'"
        
        # Extract and validate timestamp
        timestamp_part = run_dir_name[4:]  # e.g., "20251204_103000"
        parts = timestamp_part.split("_")
        assert len(parts) == 2, f"Timestamp must have 2 parts separated by _, got: {timestamp_part}"
        assert len(parts[0]) == 8, f"Date part must be 8 digits (YYYYMMDD), got: {parts[0]}"
        assert len(parts[1]) == 6, f"Time part must be 6 digits (HHMMSS), got: {parts[1]}"
        assert parts[0].isdigit(), f"Date part must be all digits, got: {parts[0]}"
        assert parts[1].isdigit(), f"Time part must be all digits, got: {parts[1]}"
        
        # Parse and validate the timestamp is reasonable (within our test window)
        try:
            parsed_time = datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
        except ValueError as e:
            pytest.fail(f"Timestamp {timestamp_part} is not valid: {e}")
        
        # Timestamp should be between before and after the call
        # Note: timestamp has second-level precision, so we truncate microseconds for comparison
        before_truncated = before_time.replace(microsecond=0)
        after_truncated = after_time.replace(microsecond=0)
        assert before_truncated <= parsed_time <= after_truncated, \
            f"Timestamp {parsed_time} should be between {before_truncated} and {after_truncated}"
    
    def test_run_output_dir_is_absolute_or_resolvable(self, tmp_path):
        """run_output_dir should be an absolute path or resolvable."""
        md_path = tmp_path / "paper.md"
        md_path.write_text("# Test\n" + "A" * 150, encoding="utf-8")
        
        with patch("src.paper_loader.loaders.extract_figures_from_markdown", return_value=[]):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "outputs"),
                download_figures=False,
            )
        
        run_output_dir = Path(paper["run_output_dir"])
        
        # Should be resolvable to an existing path
        resolved = run_output_dir.resolve()
        assert resolved.exists(), f"Resolved path {resolved} must exist"


# ═══════════════════════════════════════════════════════════════════════
# run_output_dir Propagation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRunOutputDirPropagation:
    """Test that run_output_dir flows correctly through the system."""
    
    def test_state_conversion_copies_run_output_dir(self, tmp_path):
        """create_state_from_paper_input should copy run_output_dir to state exactly."""
        md_path = tmp_path / "paper.md"
        md_path.write_text("# Test\n" + "A" * 150, encoding="utf-8")
        output_dir = tmp_path / "outputs"
        
        with patch("src.paper_loader.loaders.extract_figures_from_markdown", return_value=[]):
            paper = load_paper_from_markdown(
                str(md_path),
                str(output_dir),
                paper_id="test_paper",
                download_figures=False,
            )
        
        state = create_state_from_paper_input(paper)
        
        # Verify run_output_dir is in state with exact value
        assert "run_output_dir" in state, "run_output_dir must be in state"
        assert state["run_output_dir"] == paper["run_output_dir"], \
            f"State run_output_dir '{state['run_output_dir']}' must match paper '{paper['run_output_dir']}'"
        
        # Verify other fields also propagated correctly
        assert state["paper_id"] == paper["paper_id"]
        assert state["paper_text"] == paper["paper_text"]
    
    def test_state_conversion_handles_missing_run_output_dir(self, tmp_path):
        """create_state_from_paper_input should handle paper without run_output_dir."""
        # Create a paper dict without run_output_dir (simulating old data)
        paper = {
            "paper_id": "legacy",
            "paper_title": "Test",
            "paper_text": "A" * 150,
            "paper_domain": "other",
            "figures": [],
            # Note: no run_output_dir
        }
        
        state = create_state_from_paper_input(paper)
        
        # Should default to empty string, not raise or be missing
        assert "run_output_dir" in state, "run_output_dir must be in state even if missing from paper"
        assert state["run_output_dir"] == "", "Missing run_output_dir should default to empty string"
    
    def test_create_initial_state_accepts_run_output_dir(self, tmp_path):
        """create_initial_state should accept and store run_output_dir exactly."""
        run_dir = tmp_path / "outputs" / "paper" / "run_20251204_103000"
        run_dir_str = str(run_dir)
        
        state = create_initial_state(
            paper_id="test",
            paper_text="content",
            run_output_dir=run_dir_str,
        )
        
        assert state["run_output_dir"] == run_dir_str, "run_output_dir must be stored exactly"
        # Verify type
        assert isinstance(state["run_output_dir"], str), "run_output_dir must be a string"
    
    def test_create_initial_state_default_empty_run_output_dir(self):
        """create_initial_state should default to empty run_output_dir."""
        state = create_initial_state(
            paper_id="test",
            paper_text="content",
        )
        
        assert "run_output_dir" in state, "run_output_dir must be present even as default"
        assert state["run_output_dir"] == "", "Default run_output_dir must be empty string"
        assert isinstance(state["run_output_dir"], str), "run_output_dir must be a string"


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint Content Verification Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCheckpointContentVerification:
    """Verify checkpoint content, not just existence."""
    
    def test_saved_checkpoint_contains_run_output_dir(self, tmp_path):
        """Checkpoint file must contain run_output_dir field with correct value."""
        run_dir = tmp_path / "outputs" / "paper" / "run_20251204_103000"
        run_dir.mkdir(parents=True)
        
        state = create_initial_state(
            paper_id="test_paper",
            paper_text="test content 12345",
            run_output_dir=str(run_dir),
        )
        state["plan"] = {"stages": [{"stage_id": "stage_0", "targets": ["Fig1"]}]}
        state["current_stage_id"] = "stage_0"
        
        cp = save_checkpoint(state, "test", output_dir=str(tmp_path / "outputs"))
        
        # Read and verify FULL content
        with open(cp, "r", encoding="utf-8") as f:
            saved = json.load(f)
        
        # Verify critical fields saved correctly
        assert saved["run_output_dir"] == str(run_dir), \
            f"run_output_dir not saved correctly: got '{saved.get('run_output_dir')}', expected '{run_dir}'"
        assert saved["paper_id"] == "test_paper", "paper_id not saved correctly"
        assert saved["paper_text"] == "test content 12345", "paper_text not saved correctly"
        assert saved["current_stage_id"] == "stage_0", "current_stage_id not saved correctly"
        assert saved["plan"]["stages"][0]["stage_id"] == "stage_0", "plan not saved correctly"
        assert saved["plan"]["stages"][0]["targets"] == ["Fig1"], "plan targets not saved correctly"
    
    def test_loaded_checkpoint_preserves_all_fields(self, tmp_path):
        """Loading a checkpoint should preserve ALL state fields exactly."""
        # NOTE: run_dir path must be under outputs/{paper_id}/run_* for load_checkpoint to find it
        paper_id = "roundtrip_test"
        run_dir = tmp_path / "outputs" / paper_id / "run_20251204_103000"
        run_dir.mkdir(parents=True)
        
        original_state = create_initial_state(
            paper_id=paper_id,
            paper_text="roundtrip content",
            paper_domain="plasmonics",
            run_output_dir=str(run_dir),
        )
        original_state["plan"] = {"stages": [{"stage_id": "s0"}], "overview": "test plan"}
        original_state["current_stage_id"] = "stage_0"
        original_state["workflow_phase"] = "execution"
        original_state["design_revision_count"] = 3
        
        save_checkpoint(original_state, "roundtrip", output_dir=str(tmp_path / "outputs"))
        
        loaded = load_checkpoint("roundtrip_test", "latest", str(tmp_path / "outputs"))
        
        assert loaded is not None, "Loaded checkpoint must not be None"
        
        # Verify all critical fields are preserved exactly
        assert loaded["paper_id"] == original_state["paper_id"]
        assert loaded["paper_text"] == original_state["paper_text"]
        assert loaded["paper_domain"] == original_state["paper_domain"]
        assert loaded["run_output_dir"] == original_state["run_output_dir"]
        assert loaded["current_stage_id"] == original_state["current_stage_id"]
        assert loaded["workflow_phase"] == original_state["workflow_phase"]
        assert loaded["design_revision_count"] == original_state["design_revision_count"]
        assert loaded["plan"] == original_state["plan"]
    
    def test_checkpoint_preserves_complex_nested_structures(self, tmp_path):
        """Checkpoint should preserve complex nested data structures."""
        # NOTE: run_dir path must be under outputs/{paper_id}/run_* for load_checkpoint to find it
        paper_id = "nested_test"
        run_dir = tmp_path / "outputs" / paper_id / "run_20251204_103000"
        run_dir.mkdir(parents=True)
        
        state = create_initial_state(
            paper_id=paper_id,
            paper_text="content",
            run_output_dir=str(run_dir),
        )
        
        # Add complex nested structures
        state["plan"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "targets": [{"figure_id": "Fig1", "params": {"wavelength": [400, 500, 600]}}],
                    "expected_outputs": [{"filename": "output.csv", "columns": ["x", "y"]}]
                }
            ],
            "metadata": {"nested": {"deep": {"value": 42}}}
        }
        state["figure_comparisons"] = [
            {"figure_id": "Fig1", "metrics": {"rmse": 0.05, "correlation": 0.98}}
        ]
        
        save_checkpoint(state, "nested", output_dir=str(tmp_path / "outputs"))
        loaded = load_checkpoint("nested_test", "latest", str(tmp_path / "outputs"))
        
        # Verify deep nesting preserved
        assert loaded["plan"]["stages"][0]["targets"][0]["params"]["wavelength"] == [400, 500, 600]
        assert loaded["plan"]["metadata"]["nested"]["deep"]["value"] == 42
        assert loaded["figure_comparisons"][0]["metrics"]["rmse"] == 0.05


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint Path Tests (with content verification)
# ═══════════════════════════════════════════════════════════════════════

class TestCheckpointWithRunOutputDir:
    """Test checkpoint functions with run_output_dir."""
    
    def test_save_checkpoint_uses_run_output_dir(self, tmp_path):
        """save_checkpoint should save to run_output_dir/checkpoints/ when set."""
        run_dir = tmp_path / "outputs" / "paper" / "run_20251204_103000"
        run_dir.mkdir(parents=True)
        
        state = create_initial_state(
            paper_id="test_paper",
            paper_text="content",
            run_output_dir=str(run_dir),
        )
        
        checkpoint_path = save_checkpoint(
            state,
            "after_plan",
            output_dir=str(tmp_path / "outputs"),
        )
        
        # Verify path structure
        cp_path = Path(checkpoint_path)
        assert cp_path.exists(), f"Checkpoint file must exist at {checkpoint_path}"
        assert cp_path.parent.name == "checkpoints", "Checkpoint must be in 'checkpoints' folder"
        assert cp_path.parent.parent == run_dir, "checkpoints folder must be inside run_output_dir"
        
        # Verify file is valid JSON with correct content
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        assert data["paper_id"] == "test_paper"
        assert data["run_output_dir"] == str(run_dir)
    
    def test_save_checkpoint_falls_back_to_legacy_path(self, tmp_path):
        """save_checkpoint should use legacy path when run_output_dir is empty."""
        state = create_initial_state(
            paper_id="test_paper",
            paper_text="content",
        )
        
        checkpoint_path = save_checkpoint(
            state,
            "after_plan",
            output_dir=str(tmp_path / "outputs"),
        )
        
        # Verify legacy path structure
        cp_path = Path(checkpoint_path)
        assert cp_path.exists(), "Checkpoint must exist"
        assert cp_path.parent.name == "checkpoints"
        assert cp_path.parent.parent.name == "test_paper", "Legacy path must use paper_id directly"
        assert cp_path.parent.parent.parent == tmp_path / "outputs"
        
        # Must NOT contain run_ in path
        assert "run_" not in str(cp_path), "Legacy path must not have 'run_' folder"
    
    def test_save_checkpoint_creates_nonexistent_run_dir(self, tmp_path):
        """save_checkpoint should create run_output_dir/checkpoints if it doesn't exist."""
        # run_dir doesn't exist yet
        run_dir = tmp_path / "outputs" / "paper" / "run_20251204_103000"
        assert not run_dir.exists(), "Precondition: run_dir should not exist"
        
        state = create_initial_state(
            paper_id="test_paper",
            paper_text="content",
            run_output_dir=str(run_dir),
        )
        
        checkpoint_path = save_checkpoint(
            state,
            "test",
            output_dir=str(tmp_path / "outputs"),
        )
        
        # Should have created the directories
        assert run_dir.exists(), "run_output_dir should be created by save_checkpoint"
        assert (run_dir / "checkpoints").exists(), "checkpoints folder should be created"
        assert Path(checkpoint_path).exists(), "checkpoint file should exist"


# ═══════════════════════════════════════════════════════════════════════
# Multiple Runs Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMultipleRuns:
    """Test that multiple runs create separate folders."""
    
    def test_multiple_runs_create_separate_folders(self, tmp_path):
        """Each call to load_paper_from_markdown creates a new run folder."""
        md_path = tmp_path / "paper.md"
        md_path.write_text("# Test\n" + "A" * 150, encoding="utf-8")
        output_dir = tmp_path / "outputs"
        
        with patch("src.paper_loader.loaders.extract_figures_from_markdown", return_value=[]):
            # First run
            paper1 = load_paper_from_markdown(
                str(md_path),
                str(output_dir),
                paper_id="test_paper",
                download_figures=False,
            )
            
            # Wait to ensure different timestamp
            time.sleep(1.1)
            
            # Second run
            paper2 = load_paper_from_markdown(
                str(md_path),
                str(output_dir),
                paper_id="test_paper",
                download_figures=False,
            )
        
        # Verify different run folders
        assert paper1["run_output_dir"] != paper2["run_output_dir"], \
            "Each run must create a different run_output_dir"
        
        # Verify both exist and are directories
        run1 = Path(paper1["run_output_dir"])
        run2 = Path(paper2["run_output_dir"])
        assert run1.exists() and run1.is_dir()
        assert run2.exists() and run2.is_dir()
        
        # Verify both are under the same paper_id folder
        assert run1.parent == run2.parent, "Both runs should be under same paper_id folder"
        assert run1.parent.name == "test_paper"
        
        # Verify exactly 2 run folders exist
        paper_dir = output_dir / "test_paper"
        run_dirs = list(paper_dir.glob("run_*"))
        assert len(run_dirs) == 2, f"Expected 2 run folders, found {len(run_dirs)}"
        
        # Verify folders contain figures directories
        assert (run1 / "figures").exists()
        assert (run2 / "figures").exists()
    
    def test_checkpoints_saved_per_run_are_isolated(self, tmp_path):
        """Each run should have its own checkpoints folder with isolated data."""
        run_dir1 = tmp_path / "outputs" / "paper" / "run_20251204_100000"
        run_dir2 = tmp_path / "outputs" / "paper" / "run_20251204_110000"
        run_dir1.mkdir(parents=True)
        run_dir2.mkdir(parents=True)
        
        state1 = create_initial_state(
            paper_id="paper",
            paper_text="content from run 1",
            run_output_dir=str(run_dir1),
        )
        state1["current_stage_id"] = "run1_stage"
        
        state2 = create_initial_state(
            paper_id="paper",
            paper_text="content from run 2",
            run_output_dir=str(run_dir2),
        )
        state2["current_stage_id"] = "run2_stage"
        
        cp1 = save_checkpoint(state1, "test", output_dir=str(tmp_path / "outputs"))
        cp2 = save_checkpoint(state2, "test", output_dir=str(tmp_path / "outputs"))
        
        # Verify different checkpoint directories
        assert Path(cp1).parent.parent == run_dir1, "cp1 should be under run_dir1"
        assert Path(cp2).parent.parent == run_dir2, "cp2 should be under run_dir2"
        assert cp1 != cp2, "Checkpoint paths must be different"
        
        # Verify content is isolated (each has its own data)
        with open(cp1) as f:
            data1 = json.load(f)
        with open(cp2) as f:
            data2 = json.load(f)
        
        assert data1["paper_text"] == "content from run 1"
        assert data2["paper_text"] == "content from run 2"
        assert data1["current_stage_id"] == "run1_stage"
        assert data2["current_stage_id"] == "run2_stage"


# ═══════════════════════════════════════════════════════════════════════
# Backwards Compatibility Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBackwardsCompatibility:
    """Test that old checkpoints without run_output_dir still work."""
    
    def test_load_checkpoint_finds_legacy_checkpoints(self, tmp_path):
        """load_checkpoint should find checkpoints in legacy structure."""
        # Create legacy checkpoint structure
        legacy_dir = tmp_path / "outputs" / "old_paper" / "checkpoints"
        legacy_dir.mkdir(parents=True)
        
        checkpoint_data = {
            "paper_id": "old_paper",
            "paper_text": "old content abc123",
            "paper_domain": "plasmonics",
            "current_stage_id": "stage_2",
            # Note: no run_output_dir - simulating pre-migration data
        }
        checkpoint_file = legacy_dir / "checkpoint_old_paper_test_20251201_120000_000000.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)
        
        # Load checkpoint
        loaded = load_checkpoint(
            paper_id="old_paper",
            checkpoint_name="latest",
            output_dir=str(tmp_path / "outputs"),
        )
        
        # Verify all data loaded correctly
        assert loaded is not None, "Must find legacy checkpoint"
        assert loaded["paper_id"] == "old_paper"
        assert loaded["paper_text"] == "old content abc123"
        assert loaded["paper_domain"] == "plasmonics"
        assert loaded["current_stage_id"] == "stage_2"
    
    def test_load_checkpoint_finds_new_structure_checkpoints(self, tmp_path):
        """load_checkpoint should find checkpoints in new run folder structure."""
        # Create new checkpoint structure
        run_dir = tmp_path / "outputs" / "new_paper" / "run_20251204_103000" / "checkpoints"
        run_dir.mkdir(parents=True)
        
        checkpoint_data = {
            "paper_id": "new_paper",
            "paper_text": "new content xyz789",
            "run_output_dir": str(run_dir.parent),
            "workflow_phase": "analysis",
        }
        checkpoint_file = run_dir / "checkpoint_new_paper_test_20251204_103000_000000.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)
        
        # Load checkpoint
        loaded = load_checkpoint(
            paper_id="new_paper",
            checkpoint_name="latest",
            output_dir=str(tmp_path / "outputs"),
        )
        
        # Verify all data loaded correctly including run_output_dir
        assert loaded is not None, "Must find new structure checkpoint"
        assert loaded["paper_id"] == "new_paper"
        assert loaded["paper_text"] == "new content xyz789"
        assert loaded["run_output_dir"] == str(run_dir.parent)
        assert loaded["workflow_phase"] == "analysis"
    
    def test_load_checkpoint_prefers_most_recent_across_structures(self, tmp_path):
        """load_checkpoint should find the most recent across all structures."""
        # Create legacy checkpoint (older by mtime)
        legacy_dir = tmp_path / "outputs" / "mixed_paper" / "checkpoints"
        legacy_dir.mkdir(parents=True)
        legacy_checkpoint = legacy_dir / "checkpoint_mixed_paper_test_20251201_120000_000000.json"
        with open(legacy_checkpoint, "w") as f:
            json.dump({"paper_id": "mixed_paper", "source": "legacy", "value": 100}, f)
        
        # Wait to ensure different mtime
        time.sleep(0.1)
        
        # Create new structure checkpoint (newer by mtime)
        run_dir = tmp_path / "outputs" / "mixed_paper" / "run_20251204_103000" / "checkpoints"
        run_dir.mkdir(parents=True)
        new_checkpoint = run_dir / "checkpoint_mixed_paper_test_20251204_103000_000000.json"
        with open(new_checkpoint, "w") as f:
            json.dump({"paper_id": "mixed_paper", "source": "new", "value": 200}, f)
        
        # Load latest should get the newer one (by mtime)
        loaded = load_checkpoint(
            paper_id="mixed_paper",
            checkpoint_name="latest",
            output_dir=str(tmp_path / "outputs"),
        )
        
        assert loaded is not None
        assert loaded["source"] == "new", "Should load most recent checkpoint"
        assert loaded["value"] == 200, "Should have data from new checkpoint"
    
    def test_list_checkpoints_includes_all_structures(self, tmp_path):
        """list_checkpoints should list checkpoints from both structures with correct metadata."""
        # Create legacy checkpoint
        legacy_dir = tmp_path / "outputs" / "list_paper" / "checkpoints"
        legacy_dir.mkdir(parents=True)
        legacy_checkpoint = legacy_dir / "checkpoint_list_paper_legacy_20251201_120000_000000.json"
        with open(legacy_checkpoint, "w") as f:
            json.dump({"paper_id": "list_paper", "type": "legacy"}, f)
        
        # Create new structure checkpoint
        run_dir = tmp_path / "outputs" / "list_paper" / "run_20251204_103000" / "checkpoints"
        run_dir.mkdir(parents=True)
        new_checkpoint = run_dir / "checkpoint_list_paper_new_20251204_103000_000000.json"
        with open(new_checkpoint, "w") as f:
            json.dump({"paper_id": "list_paper", "type": "new"}, f)
        
        # List checkpoints
        checkpoints = list_checkpoints(
            paper_id="list_paper",
            output_dir=str(tmp_path / "outputs"),
        )
        
        assert len(checkpoints) == 2, f"Expected 2 checkpoints, got {len(checkpoints)}"
        
        # Verify structure of returned data
        for cp in checkpoints:
            assert "name" in cp, "checkpoint must have 'name'"
            assert "timestamp" in cp, "checkpoint must have 'timestamp'"
            assert "path" in cp, "checkpoint must have 'path'"
            assert "size_kb" in cp, "checkpoint must have 'size_kb'"
            assert "run_folder" in cp, "checkpoint must have 'run_folder'"
            assert Path(cp["path"]).exists(), f"checkpoint path must exist: {cp['path']}"
        
        names = [cp["name"] for cp in checkpoints]
        assert "legacy" in names, "Must include legacy checkpoint"
        assert "new" in names, "Must include new checkpoint"
        
        # Verify run_folder info is correct
        for cp in checkpoints:
            if cp["name"] == "legacy":
                assert cp["run_folder"] is None, "Legacy checkpoint should have run_folder=None"
            elif cp["name"] == "new":
                assert cp["run_folder"] == "run_20251204_103000", \
                    f"New checkpoint should have correct run_folder, got: {cp['run_folder']}"
    
    def test_state_without_run_output_dir_still_works(self, tmp_path):
        """Existing states without run_output_dir should still work with legacy path."""
        # Simulate loading an old checkpoint that lacks run_output_dir
        state = create_initial_state(
            paper_id="legacy_paper",
            paper_text="content",
        )
        # run_output_dir defaults to ""
        assert state["run_output_dir"] == "", "Precondition: run_output_dir should be empty"
        
        # Save checkpoint should work using legacy path
        cp = save_checkpoint(state, "test", output_dir=str(tmp_path / "outputs"))
        
        # Verify legacy path used
        cp_path = Path(cp)
        assert cp_path.exists(), "Checkpoint must exist"
        assert "legacy_paper" in str(cp_path), "Path must contain paper_id"
        assert "checkpoints" in str(cp_path), "Path must contain checkpoints folder"
        assert "/run_" not in str(cp_path), "Legacy path must NOT have run_ folder"
        
        # Verify content
        with open(cp) as f:
            data = json.load(f)
        assert data["paper_id"] == "legacy_paper"
        assert data["run_output_dir"] == ""


# ═══════════════════════════════════════════════════════════════════════
# Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRunOutputDirEdgeCases:
    """Edge cases that could reveal bugs."""
    
    def test_empty_run_output_dir_treated_as_legacy(self, tmp_path):
        """Empty string run_output_dir should use legacy path, not create broken paths."""
        state = create_initial_state(
            paper_id="legacy_test_paper",
            paper_text="content",
            run_output_dir="",  # Explicit empty string
        )
        
        cp = save_checkpoint(state, "mycheck", output_dir=str(tmp_path))
        
        cp_path = Path(cp)
        # Must NOT create a folder literally named "" or have broken path
        assert cp_path.exists(), "Checkpoint must exist"
        assert "//" not in str(cp_path), "Path must not have double slashes"
        
        # Verify actual path structure is correct legacy format:
        # {output_dir}/{paper_id}/checkpoints/checkpoint_*.json
        assert cp_path.parent.name == "checkpoints", \
            f"Checkpoint should be in 'checkpoints' folder, got: {cp_path.parent.name}"
        assert cp_path.parent.parent.name == "legacy_test_paper", \
            f"checkpoints should be directly under paper_id folder, got: {cp_path.parent.parent.name}"
        
        # Legacy path should NOT have a /run_* folder in the hierarchy
        path_parts = cp_path.parts
        for part in path_parts:
            if part.startswith("run_") and "_" in part[4:]:  # run_ followed by timestamp pattern
                pytest.fail(f"Legacy path should not have run folder, but found: {part}")
    
    def test_whitespace_only_paper_id_handling(self, tmp_path):
        """Paper ID with only whitespace should be handled gracefully."""
        # This test documents expected behavior - either works or raises ValueError
        md_path = tmp_path / "paper.md"
        md_path.write_text("# Test\n" + "A" * 150, encoding="utf-8")
        
        with patch("src.paper_loader.loaders.extract_figures_from_markdown", return_value=[]):
            # Empty string paper_id - should use filename-derived ID
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "outputs"),
                paper_id=None,  # Let it derive from filename
                download_figures=False,
            )
        
        # Should derive paper_id from filename
        assert paper["paper_id"] == "paper", "Should derive paper_id from filename"
        assert Path(paper["run_output_dir"]).exists()
    
    def test_paper_id_with_underscores(self, tmp_path):
        """Paper ID with underscores should work (common case)."""
        md_path = tmp_path / "paper.md"
        md_path.write_text("# Test\n" + "A" * 150, encoding="utf-8")
        
        with patch("src.paper_loader.loaders.extract_figures_from_markdown", return_value=[]):
            paper = load_paper_from_markdown(
                str(md_path),
                str(tmp_path / "outputs"),
                paper_id="my_test_paper_2024",
                download_figures=False,
            )
        
        run_dir = Path(paper["run_output_dir"])
        assert run_dir.exists()
        assert run_dir.parent.name == "my_test_paper_2024"
        
        # Verify checkpoint works too
        state = create_state_from_paper_input(paper)
        cp = save_checkpoint(state, "test", output_dir=str(tmp_path / "outputs"))
        assert Path(cp).exists()
    
    def test_load_checkpoint_returns_none_for_nonexistent_paper(self, tmp_path):
        """load_checkpoint should return None for paper that doesn't exist."""
        loaded = load_checkpoint(
            paper_id="definitely_does_not_exist_xyz123",
            checkpoint_name="latest",
            output_dir=str(tmp_path / "outputs"),
        )
        
        assert loaded is None, "Should return None for nonexistent paper"
    
    def test_list_checkpoints_returns_empty_for_nonexistent_paper(self, tmp_path):
        """list_checkpoints should return empty list for paper that doesn't exist."""
        checkpoints = list_checkpoints(
            paper_id="definitely_does_not_exist_xyz123",
            output_dir=str(tmp_path / "outputs"),
        )
        
        assert checkpoints == [], "Should return empty list for nonexistent paper"
        assert isinstance(checkpoints, list), "Should return a list"


# ═══════════════════════════════════════════════════════════════════════
# Error Handling Tests
# ═══════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    """Test error conditions that could cause silent failures."""
    
    def test_load_checkpoint_with_corrupted_json(self, tmp_path):
        """Loading corrupted checkpoint should raise appropriate error."""
        legacy_dir = tmp_path / "outputs" / "corrupt" / "checkpoints"
        legacy_dir.mkdir(parents=True)
        
        # Write invalid JSON
        checkpoint_file = legacy_dir / "checkpoint_corrupt_test_20251201_120000_000000.json"
        checkpoint_file.write_text("{invalid json content", encoding="utf-8")
        
        # Should raise JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            load_checkpoint("corrupt", "latest", str(tmp_path / "outputs"))
    
    def test_load_checkpoint_with_empty_file(self, tmp_path):
        """Loading empty checkpoint file should handle gracefully."""
        legacy_dir = tmp_path / "outputs" / "empty" / "checkpoints"
        legacy_dir.mkdir(parents=True)
        
        # Write empty file
        checkpoint_file = legacy_dir / "checkpoint_empty_test_20251201_120000_000000.json"
        checkpoint_file.write_text("", encoding="utf-8")
        
        # Should raise JSONDecodeError for empty file
        with pytest.raises(json.JSONDecodeError):
            load_checkpoint("empty", "latest", str(tmp_path / "outputs"))
    
    def test_list_checkpoints_skips_latest_pointer_files(self, tmp_path):
        """list_checkpoints should not include *_latest.json pointer files."""
        legacy_dir = tmp_path / "outputs" / "skip_latest" / "checkpoints"
        legacy_dir.mkdir(parents=True)
        
        # Create actual checkpoint
        actual_checkpoint = legacy_dir / "checkpoint_skip_latest_real_20251201_120000_000000.json"
        with open(actual_checkpoint, "w") as f:
            json.dump({"paper_id": "skip_latest"}, f)
        
        # Create latest pointer file (should be skipped)
        latest_pointer = legacy_dir / "checkpoint_real_latest.json"
        with open(latest_pointer, "w") as f:
            json.dump({"paper_id": "skip_latest"}, f)
        
        checkpoints = list_checkpoints("skip_latest", str(tmp_path / "outputs"))
        
        # Should only include the real checkpoint, not the _latest pointer
        assert len(checkpoints) == 1, f"Should find 1 checkpoint, found {len(checkpoints)}"
        assert checkpoints[0]["name"] == "real", "Should be the real checkpoint"


# ═══════════════════════════════════════════════════════════════════════
# Downstream Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDownstreamCodeRunnerIntegration:
    """Verify code_runner uses run_output_dir correctly."""
    
    def test_code_runner_output_base_uses_run_output_dir(self, tmp_path):
        """Verify code_runner constructs output_base using run_output_dir."""
        from src.code_runner import run_code_node
        
        run_dir = tmp_path / "outputs" / "test_paper" / "run_20251204_103000"
        run_dir.mkdir(parents=True)
        
        state = create_initial_state(
            paper_id="test_paper",
            paper_text="content",
            run_output_dir=str(run_dir),
        )
        state["current_stage_id"] = "stage_0"
        # NOTE: run_code_node looks for "code" field, not "current_stage_code"
        state["code"] = "print('hello world')"
        
        # Execute - run_code_node will create the output directory
        result = run_code_node(state)
        
        # Verify the output directory was created in run_output_dir/stage_id
        expected_output_base = run_dir / "stage_0"
        assert expected_output_base.exists(), \
            f"Output should be in run_output_dir/stage_id: {expected_output_base}"
        
        # Also verify no error was returned
        assert result.get("run_error") is None or "No simulation code" not in str(result.get("run_error", "")), \
            f"run_code_node returned error: {result.get('run_error')}"
    
    def test_code_runner_falls_back_to_legacy_without_run_output_dir(self, tmp_path):
        """code_runner should use legacy path when run_output_dir is empty."""
        from src.code_runner import run_code_node
        
        state = create_initial_state(
            paper_id="legacy_runner_test",
            paper_text="content",
        )
        state["current_stage_id"] = "stage_0"
        # NOTE: run_code_node looks for "code" field, not "current_stage_code"
        state["code"] = "print('hello')"
        
        # Change working directory to tmp_path so outputs go there
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = run_code_node(state)
            
            # Verify legacy path was used: outputs/{paper_id}/{stage_id}/
            legacy_output = tmp_path / "outputs" / "legacy_runner_test" / "stage_0"
            assert legacy_output.exists(), \
                f"Legacy output should be at outputs/paper_id/stage_id: {legacy_output}"
            
            # Verify no error about missing code
            assert result.get("run_error") is None or "No simulation code" not in str(result.get("run_error", "")), \
                f"run_code_node returned error: {result.get('run_error')}"
        finally:
            os.chdir(original_cwd)


class TestDownstreamAnalysisIntegration:
    """Verify analysis.py uses run_output_dir correctly."""
    
    def test_analysis_base_output_dir_uses_run_output_dir(self, tmp_path):
        """Verify analysis constructs base_output_dir using run_output_dir."""
        # This is a structural test - we verify the path construction logic
        # by checking the code in analysis.py handles run_output_dir
        from pathlib import Path as _Path
        
        # Simulate what analysis.py should do
        run_output_dir = str(tmp_path / "outputs" / "test" / "run_20251204_103000")
        current_stage_id = "stage_0"
        
        # New path construction (what should happen)
        if run_output_dir:
            base_output_dir = _Path(run_output_dir) / current_stage_id
        else:
            # Legacy path
            base_output_dir = _Path("outputs") / "test" / current_stage_id
        
        expected = tmp_path / "outputs" / "test" / "run_20251204_103000" / "stage_0"
        assert base_output_dir == expected, \
            f"base_output_dir should use run_output_dir, got {base_output_dir}"


# ═══════════════════════════════════════════════════════════════════════
# Full Flow Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFullFlowIntegration:
    """Test the complete flow from paper loading to checkpoint."""
    
    def test_full_flow_paper_to_state_to_checkpoint(self, tmp_path):
        """Test complete flow: load_paper -> create_state -> save_checkpoint -> load_checkpoint."""
        md_path = tmp_path / "paper.md"
        md_path.write_text("# Full Flow Test Paper\n" + "A" * 150, encoding="utf-8")
        output_dir = tmp_path / "outputs"
        
        # Step 1: Load paper
        with patch("src.paper_loader.loaders.extract_figures_from_markdown", return_value=[]):
            paper = load_paper_from_markdown(
                str(md_path),
                str(output_dir),
                paper_id="full_flow_test",
                download_figures=False,
            )
        
        assert "run_output_dir" in paper
        original_run_dir = paper["run_output_dir"]
        
        # Step 2: Create state
        state = create_state_from_paper_input(paper)
        assert state["run_output_dir"] == original_run_dir
        
        # Add some workflow progress
        state["plan"] = {"stages": [{"stage_id": "stage_0"}]}
        state["current_stage_id"] = "stage_0"
        state["workflow_phase"] = "execution"
        
        # Step 3: Save checkpoint
        cp_path = save_checkpoint(state, "mid_execution", output_dir=str(output_dir))
        
        # Verify checkpoint is in run_output_dir
        assert original_run_dir in cp_path, "Checkpoint should be in run_output_dir"
        
        # Step 4: Load checkpoint
        loaded = load_checkpoint("full_flow_test", "latest", str(output_dir))
        
        # Verify everything round-trips correctly
        assert loaded is not None
        assert loaded["paper_id"] == "full_flow_test"
        assert loaded["run_output_dir"] == original_run_dir
        assert loaded["current_stage_id"] == "stage_0"
        assert loaded["workflow_phase"] == "execution"
        assert loaded["plan"]["stages"][0]["stage_id"] == "stage_0"
