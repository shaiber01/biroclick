"""Tests for stage_type validation handling."""

from unittest.mock import patch

import pytest

from src.agents.stage_selection import select_stage_node
from tests.agents.stage_selection.utils import create_stage

class TestStageTypeValidation:
    """Tests for stage_type validation."""

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_blocks_stage_without_stage_type(self, mock_update):
        """Should block stage without stage_type."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "status": "not_started", "dependencies": []},
                ]
            },
        }
        
        result = select_stage_node(state)
        
        mock_update.assert_called()
        assert result["current_stage_id"] is None

    @patch("src.agents.stage_selection.update_progress_stage_status")
    def test_blocks_stage_with_unknown_stage_type(self, mock_update):
        """Should block stage with unknown stage_type."""
        state = {
            "plan": {"stages": []},
            "progress": {
                "stages": [
                    create_stage("stage0", "UNKNOWN_TYPE", "not_started"),
                ]
            },
        }
        
        result = select_stage_node(state)
        
        mock_update.assert_called()
        assert result["current_stage_id"] is None
