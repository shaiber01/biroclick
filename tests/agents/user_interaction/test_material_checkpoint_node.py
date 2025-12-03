"""Tests for material_checkpoint_node."""

from unittest.mock import patch

import pytest

from src.agents.user_interaction import material_checkpoint_node

class TestMaterialCheckpointNode:
    """Tests for material_checkpoint_node function."""

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_triggers_material_validation(self, mock_format, mock_extract):
        """Should trigger material validation checkpoint."""
        mock_extract.return_value = [
            {"material_id": "gold", "name": "Gold", "path": "gold.csv"}
        ]
        mock_format.return_value = "Material checkpoint question"
        
        state = {
            "current_stage_id": "stage0_material_validation",
            "stage_outputs": {"files": ["gold.csv"]},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "material_checkpoint"
        assert "pending_validated_materials" in result
        # Check that extracted materials are passed to the result
        assert result["pending_validated_materials"] == mock_extract.return_value

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_includes_pending_validated_materials(self, mock_format, mock_extract):
        """Should include pending validated materials in result."""
        mock_extract.return_value = [
            {"material_id": "gold", "name": "Gold"},
            {"material_id": "silver", "name": "Silver"},
        ]
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert len(result["pending_validated_materials"]) == 2
        assert result["pending_validated_materials"][0]["material_id"] == "gold"

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_handles_no_materials(self, mock_format, mock_extract):
        """Should handle case with no materials detected."""
        mock_extract.return_value = []
        mock_format.return_value = "No materials detected"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["pending_validated_materials"] == []
        # Should include warning about no materials in the question text
        assert "WARNING" in result["pending_user_questions"][0]

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_extracts_plot_files(self, mock_format, mock_extract):
        """Should extract plot files from stage outputs."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {
                "files": ["data.csv", "plot.png", "comparison.pdf", "results.jpg"],
            },
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result["awaiting_user_input"] is True
        # Format function should be called with plot files
        call_args = mock_format.call_args
        plot_files = call_args[0][2]  # Third argument is plot_files
        assert "plot.png" in plot_files
        assert "comparison.pdf" in plot_files
        assert "results.jpg" in plot_files
        assert "data.csv" not in plot_files

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_finds_material_validation_stage(self, mock_format, mock_extract):
        """Should find MATERIAL_VALIDATION stage in progress."""
        mock_extract.return_value = [{"material_id": "gold"}]
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "summary": "Validated gold"},
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE"},
                ]
            },
        }
        
        result = material_checkpoint_node(state)
        
        # Should pass stage0_info to format function
        call_args = mock_format.call_args
        stage_info = call_args[0][1]  # Second argument is stage_info
        assert stage_info["stage_type"] == "MATERIAL_VALIDATION"

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_handles_empty_progress(self, mock_format, mock_extract):
        """Should handle empty progress gracefully."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {},  # Empty progress
        }
        
        result = material_checkpoint_node(state)
        
        assert result["awaiting_user_input"] is True

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_sets_workflow_phase(self, mock_format, mock_extract):
        """Should set workflow_phase to material_checkpoint."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result["workflow_phase"] == "material_checkpoint"

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_sets_last_node_before_ask_user(self, mock_format, mock_extract):
        """Should set last_node_before_ask_user."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result["last_node_before_ask_user"] == "material_checkpoint"

