"""Tests for material_checkpoint_node."""

from unittest.mock import patch, MagicMock

import pytest

from src.agents.user_interaction import material_checkpoint_node


class TestMaterialCheckpointNode:
    """Tests for material_checkpoint_node function."""

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_triggers_material_validation(self, mock_format, mock_extract):
        """Should trigger material validation checkpoint with all required fields."""
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
        
        # Verify all required return fields are present and correct
        assert result.get("ask_user_trigger") is not None
        assert result["ask_user_trigger"] == "material_checkpoint"
        assert result["workflow_phase"] == "material_checkpoint"
        assert result["last_node_before_ask_user"] == "material_checkpoint"
        assert "pending_validated_materials" in result
        assert "pending_user_questions" in result
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) == 1
        assert result["pending_user_questions"][0] == "Material checkpoint question"
        # Check that extracted materials are passed to the result exactly
        assert result["pending_validated_materials"] == mock_extract.return_value
        assert len(result["pending_validated_materials"]) == 1
        assert result["pending_validated_materials"][0]["material_id"] == "gold"
        
        # Verify function was called with correct state
        mock_extract.assert_called_once_with(state)
        mock_format.assert_called_once()
        format_call_args = mock_format.call_args[0]
        assert format_call_args[0] == state  # First arg is state
        assert format_call_args[1] is None  # stage0_info is None when no stages found
        assert isinstance(format_call_args[2], list)  # plot_files is a list

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_includes_pending_validated_materials(self, mock_format, mock_extract):
        """Should include pending validated materials in result with exact values."""
        mock_extract.return_value = [
            {"material_id": "gold", "name": "Gold", "path": "gold.csv"},
            {"material_id": "silver", "name": "Silver", "path": "silver.csv"},
        ]
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        # Verify exact material count and values
        assert len(result["pending_validated_materials"]) == 2
        assert result["pending_validated_materials"][0]["material_id"] == "gold"
        assert result["pending_validated_materials"][0]["name"] == "Gold"
        assert result["pending_validated_materials"][0]["path"] == "gold.csv"
        assert result["pending_validated_materials"][1]["material_id"] == "silver"
        assert result["pending_validated_materials"][1]["name"] == "Silver"
        assert result["pending_validated_materials"][1]["path"] == "silver.csv"
        # Verify materials are passed exactly as returned by extract
        assert result["pending_validated_materials"] == mock_extract.return_value

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_handles_no_materials(self, mock_format, mock_extract):
        """Should handle case with no materials detected and include warning."""
        mock_extract.return_value = []
        mock_format.return_value = "No materials detected"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result.get("ask_user_trigger") is not None
        assert result["pending_validated_materials"] == []
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) == 1
        # Should include warning about no materials in the question text
        question_text = result["pending_user_questions"][0]
        assert "WARNING" in question_text
        assert "No materials were automatically detected" in question_text
        assert "CHANGE_MATERIAL" in question_text
        assert "CHANGE_DATABASE" in question_text
        # Verify warning is appended to formatted question
        assert question_text.endswith("CHANGE_DATABASE' to specify them manually.")
        # Verify format was called first, then warning appended
        assert question_text.startswith("No materials detected")

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_extracts_plot_files(self, mock_format, mock_extract):
        """Should extract plot files from stage outputs correctly."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {
                "files": ["data.csv", "plot.png", "comparison.pdf", "results.jpg", "text.txt"],
            },
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result.get("ask_user_trigger") is not None
        # Format function should be called with plot files
        call_args = mock_format.call_args[0]
        plot_files = call_args[2]  # Third argument is plot_files
        assert isinstance(plot_files, list)
        assert len(plot_files) == 3  # Only image files
        assert "plot.png" in plot_files
        assert "comparison.pdf" in plot_files
        assert "results.jpg" in plot_files
        # Verify non-image files are excluded
        assert "data.csv" not in plot_files
        assert "text.txt" not in plot_files
        # Verify order is preserved
        assert plot_files == ["plot.png", "comparison.pdf", "results.jpg"]

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_extracts_plot_files_case_insensitive(self, mock_format, mock_extract):
        """Should extract plot files with case-insensitive extension matching."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {
                "files": ["PLOT.PNG", "comparison.PDF", "results.JPG", "data.CSV"],
            },
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        call_args = mock_format.call_args[0]
        plot_files = call_args[2]
        assert len(plot_files) == 3
        assert "PLOT.PNG" in plot_files
        assert "comparison.PDF" in plot_files
        assert "results.JPG" in plot_files
        assert "data.CSV" not in plot_files

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_finds_material_validation_stage(self, mock_format, mock_extract):
        """Should find MATERIAL_VALIDATION stage in progress and pass to format."""
        mock_extract.return_value = [{"material_id": "gold"}]
        mock_format.return_value = "Question"
        
        stage0_info = {
            "stage_id": "stage0",
            "stage_type": "MATERIAL_VALIDATION",
            "summary": "Validated gold",
            "status": "completed"
        }
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {
                "stages": [
                    stage0_info,
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE"},
                ]
            },
        }
        
        result = material_checkpoint_node(state)
        
        # Should pass stage0_info to format function
        call_args = mock_format.call_args[0]
        stage_info = call_args[1]  # Second argument is stage_info
        assert stage_info is not None
        assert stage_info["stage_type"] == "MATERIAL_VALIDATION"
        assert stage_info["stage_id"] == "stage0"
        assert stage_info["summary"] == "Validated gold"
        assert stage_info["status"] == "completed"
        # Verify it's the exact same dict object
        assert stage_info == stage0_info

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_finds_first_material_validation_stage(self, mock_format, mock_extract):
        """Should find first MATERIAL_VALIDATION stage when multiple exist."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        first_stage = {"stage_id": "stage0", "stage_type": "MATERIAL_VALIDATION", "summary": "First"}
        second_stage = {"stage_id": "stage0b", "stage_type": "MATERIAL_VALIDATION", "summary": "Second"}
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {
                "stages": [
                    first_stage,
                    second_stage,
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE"},
                ]
            },
        }
        
        result = material_checkpoint_node(state)
        
        # Should find the first MATERIAL_VALIDATION stage
        call_args = mock_format.call_args[0]
        stage_info = call_args[1]
        assert stage_info == first_stage
        assert stage_info["summary"] == "First"

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_handles_no_material_validation_stage(self, mock_format, mock_extract):
        """Should handle case when no MATERIAL_VALIDATION stage exists."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "stage_type": "SINGLE_STRUCTURE"},
                    {"stage_id": "stage2", "stage_type": "ANALYSIS"},
                ]
            },
        }
        
        result = material_checkpoint_node(state)
        
        # Should pass None as stage0_info when no MATERIAL_VALIDATION stage found
        call_args = mock_format.call_args[0]
        stage_info = call_args[1]
        assert stage_info is None

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
        
        assert result.get("ask_user_trigger") is not None
        # Should pass None as stage0_info
        call_args = mock_format.call_args[0]
        stage_info = call_args[1]
        assert stage_info is None

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_handles_missing_progress(self, mock_format, mock_extract):
        """Should handle missing progress key."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            # progress key missing
        }
        
        result = material_checkpoint_node(state)
        
        assert result.get("ask_user_trigger") is not None
        # Should pass None as stage0_info
        call_args = mock_format.call_args[0]
        stage_info = call_args[1]
        assert stage_info is None

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_handles_missing_stage_outputs(self, mock_format, mock_extract):
        """Should handle missing stage_outputs key."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "progress": {"stages": []},
            # stage_outputs key missing
        }
        
        result = material_checkpoint_node(state)
        
        assert result.get("ask_user_trigger") is not None
        # Should pass empty list as plot_files
        call_args = mock_format.call_args[0]
        plot_files = call_args[2]
        assert plot_files == []

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_handles_empty_stage_outputs(self, mock_format, mock_extract):
        """Should handle empty stage_outputs dict."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {},  # Empty dict
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result.get("ask_user_trigger") is not None
        # Should pass empty list as plot_files
        call_args = mock_format.call_args[0]
        plot_files = call_args[2]
        assert plot_files == []

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_handles_missing_files_key(self, mock_format, mock_extract):
        """Should handle missing files key in stage_outputs."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"other_key": "value"},  # files key missing
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result.get("ask_user_trigger") is not None
        # Should pass empty list as plot_files
        call_args = mock_format.call_args[0]
        plot_files = call_args[2]
        assert plot_files == []

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_handles_none_files(self, mock_format, mock_extract):
        """Should handle None value for files key."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": None},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        # Should handle None gracefully - list comprehension will fail if files is None
        # This test will reveal if the code handles None properly
        call_args = mock_format.call_args[0]
        plot_files = call_args[2]
        # If code doesn't handle None, this will fail
        assert isinstance(plot_files, list)

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_handles_empty_stages_list(self, mock_format, mock_extract):
        """Should handle empty stages list."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},  # Empty list
        }
        
        result = material_checkpoint_node(state)
        
        assert result.get("ask_user_trigger") is not None
        # Should pass None as stage0_info
        call_args = mock_format.call_args[0]
        stage_info = call_args[1]
        assert stage_info is None

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_handles_none_stages(self, mock_format, mock_extract):
        """Should handle None value for stages key."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": None},  # None value
        }
        
        result = material_checkpoint_node(state)
        
        # Should handle None gracefully - iteration will fail if stages is None
        # This test will reveal if the code handles None properly
        call_args = mock_format.call_args[0]
        stage_info = call_args[1]
        # If code doesn't handle None, this will fail
        assert stage_info is None or isinstance(stage_info, dict)

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_sets_workflow_phase(self, mock_format, mock_extract):
        """Should set workflow_phase to material_checkpoint exactly."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result["workflow_phase"] == "material_checkpoint"
        assert isinstance(result["workflow_phase"], str)

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_sets_last_node_before_ask_user(self, mock_format, mock_extract):
        """Should set last_node_before_ask_user to material_checkpoint."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert result["last_node_before_ask_user"] == "material_checkpoint"
        assert isinstance(result["last_node_before_ask_user"], str)

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_pending_user_questions_is_list(self, mock_format, mock_extract):
        """Should set pending_user_questions as a list with exactly one question."""
        mock_extract.return_value = []
        mock_format.return_value = "Formatted question text"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        assert isinstance(result["pending_user_questions"], list)
        assert len(result["pending_user_questions"]) == 1
        # When no materials, warning is appended to question
        question = result["pending_user_questions"][0]
        assert question.startswith("Formatted question text")
        assert "WARNING" in question

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_warning_appended_when_no_materials(self, mock_format, mock_extract):
        """Should append warning to question when no materials detected."""
        mock_extract.return_value = []  # No materials
        mock_format.return_value = "Base question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        question = result["pending_user_questions"][0]
        # Should start with formatted question
        assert question.startswith("Base question")
        # Should end with warning
        assert question.endswith("CHANGE_DATABASE' to specify them manually.")
        # Should contain warning text
        assert "⚠️ WARNING" in question
        assert "No materials were automatically detected" in question
        assert "Code generation will FAIL" in question

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_no_warning_when_materials_exist(self, mock_format, mock_extract):
        """Should NOT append warning when materials are detected."""
        mock_extract.return_value = [{"material_id": "gold"}]  # Materials exist
        mock_format.return_value = "Base question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        question = result["pending_user_questions"][0]
        # Should be exactly the formatted question, no warning appended
        assert question == "Base question"
        assert "WARNING" not in question
        assert "No materials were automatically detected" not in question

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_all_return_keys_present(self, mock_format, mock_extract):
        """Should return all required keys in result dict."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        # Verify all required keys are present
        required_keys = {
            "workflow_phase",
            "pending_user_questions",
            "awaiting_user_input",
            "ask_user_trigger",
            "last_node_before_ask_user",
            "pending_validated_materials",
        }
        assert set(result.keys()) == required_keys
        # Verify no extra keys
        assert len(result) == len(required_keys)

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_extract_materials_called_with_state(self, mock_format, mock_extract):
        """Should call extract_validated_materials with the state."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        material_checkpoint_node(state)
        
        # Verify extract was called with state
        mock_extract.assert_called_once_with(state)

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_format_called_with_correct_arguments(self, mock_format, mock_extract):
        """Should call format_material_checkpoint_question with correct arguments."""
        mock_extract.return_value = [{"material_id": "gold"}]
        mock_format.return_value = "Question"
        
        stage0_info = {"stage_type": "MATERIAL_VALIDATION"}
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": ["plot.png"]},
            "progress": {"stages": [stage0_info]},
        }
        
        material_checkpoint_node(state)
        
        # Verify format was called with correct arguments
        mock_format.assert_called_once()
        call_args = mock_format.call_args[0]
        assert call_args[0] == state  # state
        assert call_args[1] == stage0_info  # stage0_info
        assert call_args[2] == ["plot.png"]  # plot_files
        assert call_args[3] == [{"material_id": "gold"}]  # pending_materials

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_stage_type_matching_is_exact(self, mock_format, mock_extract):
        """Should match stage_type exactly, not partial matches."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {
                "stages": [
                    {"stage_type": "MATERIAL_VALIDATION_EXTENDED"},  # Not exact match
                    {"stage_type": "PRE_MATERIAL_VALIDATION"},  # Not exact match
                    {"stage_type": "MATERIAL_VALIDATION"},  # Exact match
                ]
            },
        }
        
        result = material_checkpoint_node(state)
        
        # Should find the exact match
        call_args = mock_format.call_args[0]
        stage_info = call_args[1]
        assert stage_info is not None
        assert stage_info["stage_type"] == "MATERIAL_VALIDATION"

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_plot_files_extraction_preserves_order(self, mock_format, mock_extract):
        """Should preserve order of plot files from stage_outputs."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        files = ["first.png", "second.pdf", "third.jpg", "fourth.png"]
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": files},
            "progress": {"stages": []},
        }
        
        material_checkpoint_node(state)
        
        call_args = mock_format.call_args[0]
        plot_files = call_args[2]
        # Should preserve order
        assert plot_files == ["first.png", "second.pdf", "third.jpg", "fourth.png"]

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_handles_non_list_files(self, mock_format, mock_extract):
        """Should handle files that is not a list (e.g., string, dict)."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": "not_a_list"},  # Wrong type
            "progress": {"stages": []},
        }
        
        # This should reveal if code handles non-list files properly
        # If code doesn't handle this, it will raise TypeError
        try:
            result = material_checkpoint_node(state)
            call_args = mock_format.call_args[0]
            plot_files = call_args[2]
            # If it doesn't crash, verify it's a list
            assert isinstance(plot_files, list)
        except (TypeError, AttributeError):
            # If it crashes, that's a bug in the component
            pytest.fail("material_checkpoint_node should handle non-list files gracefully")

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_handles_non_list_stages(self, mock_format, mock_extract):
        """Should handle stages that is not a list."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": "not_a_list"},  # Wrong type
        }
        
        # This should reveal if code handles non-list stages properly
        try:
            result = material_checkpoint_node(state)
            call_args = mock_format.call_args[0]
            stage_info = call_args[1]
            # If it doesn't crash, verify stage_info is None or dict
            assert stage_info is None or isinstance(stage_info, dict)
        except (TypeError, AttributeError):
            # If it crashes, that's a bug in the component
            pytest.fail("material_checkpoint_node should handle non-list stages gracefully")

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_handles_stage_without_stage_type(self, mock_format, mock_extract):
        """Should handle stages without stage_type key."""
        mock_extract.return_value = []
        mock_format.return_value = "Question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {
                "stages": [
                    {"stage_id": "stage0", "summary": "No stage_type"},  # Missing stage_type
                    {"stage_id": "stage1", "stage_type": "MATERIAL_VALIDATION"},
                ]
            },
        }
        
        result = material_checkpoint_node(state)
        
        # Should find the stage with stage_type
        call_args = mock_format.call_args[0]
        stage_info = call_args[1]
        assert stage_info is not None
        assert stage_info["stage_type"] == "MATERIAL_VALIDATION"

    @patch("src.agents.user_interaction.extract_validated_materials")
    def test_integration_with_extract_validated_materials(self, mock_extract):
        """Integration test: should work with actual extract_validated_materials."""
        # Don't mock format, but we need to verify it's called correctly
        from src.agents.user_interaction import format_material_checkpoint_question
        
        mock_extract.return_value = [
            {"material_id": "gold", "name": "Gold", "path": "materials/gold.csv"}
        ]
        
        state = {
            "paper_id": "test_paper",
            "current_stage_id": "stage0",
            "stage_outputs": {"files": ["plot.png"]},
            "progress": {
                "stages": [
                    {"stage_type": "MATERIAL_VALIDATION", "summary": "Validated"}
                ]
            },
        }
        
        result = material_checkpoint_node(state)
        
        # Verify extract was called
        mock_extract.assert_called_once_with(state)
        # Verify result contains materials
        assert result["pending_validated_materials"] == mock_extract.return_value
        # Verify format was called (not mocked, so it will use real function)
        # We can't easily verify this without patching, but we can verify the result
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) == 1

    @patch("src.agents.user_interaction.extract_validated_materials")
    @patch("src.agents.user_interaction.format_material_checkpoint_question")
    def test_warning_message_exact_content(self, mock_format, mock_extract):
        """Should include exact warning message when no materials."""
        mock_extract.return_value = []
        mock_format.return_value = "Base question"
        
        state = {
            "current_stage_id": "stage0",
            "stage_outputs": {"files": []},
            "progress": {"stages": []},
        }
        
        result = material_checkpoint_node(state)
        
        question = result["pending_user_questions"][0]
        # Verify exact warning message content
        expected_warning = (
            "\n\n⚠️ WARNING: No materials were automatically detected! "
            "Code generation will FAIL without materials. "
            "Please select 'CHANGE_MATERIAL' or 'CHANGE_DATABASE' to specify them manually."
        )
        assert expected_warning in question
        assert question == "Base question" + expected_warning
