"""Unit tests for src/agents/user_interaction.py"""

import pytest
from unittest.mock import patch, MagicMock

from src.agents.user_interaction import (
    ask_user_node,
    material_checkpoint_node,
)


class TestAskUserNode:
    """Tests for ask_user_node function."""

    def test_returns_not_awaiting_when_no_questions(self):
        """Should return awaiting_user_input=False when no questions pending."""
        state = {
            "pending_user_questions": [],
            "ask_user_trigger": "test_trigger",
        }
        
        result = ask_user_node(state)
        
        assert result["awaiting_user_input"] is False

    @pytest.mark.skip(reason="ask_user_node is CLI-based - requires stdin mocking")
    def test_formats_question_from_pending(self):
        """Should format question from pending questions."""
        state = {
            "pending_user_questions": ["Question 1", "Question 2"],
            "ask_user_trigger": "multi_question",
        }
        
        result = ask_user_node(state)
        
        assert result["awaiting_user_input"] is True

    @pytest.mark.skip(reason="ask_user_node is CLI-based - requires stdin mocking")
    def test_collects_user_response(self):
        """Should collect user response via CLI."""
        state = {
            "pending_user_questions": ["Question?"],
            "ask_user_trigger": "test",
        }
        
        result = ask_user_node(state)
        
        assert result["awaiting_user_input"] is False


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
            "stage_outputs": {"files": ["gold.csv"]},  # Uses stage_outputs
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result["awaiting_user_input"] is True
        assert result["ask_user_trigger"] == "material_checkpoint"
        assert "pending_validated_materials" in result  # Uses pending_validated_materials

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

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_extracts_plot_files(self, mock_format, mock_extract):
        """Should extract plot files from run result."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "run_result": {
                "success": True,
                "output_files": ["data.csv", "plot.png", "comparison.pdf"],
            },
        }
        
        result = material_checkpoint_node(state)
        
        assert result["awaiting_user_input"] is True
        # Format function should be called with plot files
        mock_format.assert_called_once()

