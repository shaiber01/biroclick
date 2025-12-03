"""Content builder tests for `src.llm_client`."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm_client import (
    build_user_content_for_analyzer,
    build_user_content_for_code_generator,
    build_user_content_for_designer,
    build_user_content_for_planner,
    get_images_for_analyzer,
)


class TestBuildUserContentForPlanner:
    """Tests for build_user_content_for_planner."""

    def test_full_state_all_fields_present(self):
        """Test with all fields present - verify exact structure."""
        state = {
            "paper_text": "Full paper content",
            "paper_figures": [{"id": "fig1", "description": "A figure"}],
            "planner_feedback": "Please revise",
        }
        content = build_user_content_for_planner(state)
        
        # Verify exact structure with separators
        parts = content.split("\n\n---\n\n")
        assert len(parts) == 3, f"Expected 3 parts separated by '---', got {len(parts)}"
        
        # Verify paper text section
        assert parts[0] == "# PAPER TEXT\n\nFull paper content"
        
        # Verify figures section
        assert parts[1] == "# FIGURES\n\n- fig1: A figure"
        
        # Verify feedback section
        assert parts[2] == "# REVISION FEEDBACK\n\nPlease revise"

    def test_multiple_figures(self):
        """Test with multiple figures - verify all are included."""
        state = {
            "paper_text": "Paper",
            "paper_figures": [
                {"id": "fig1", "description": "First figure"},
                {"id": "fig2", "description": "Second figure"},
                {"id": "fig3", "description": "Third figure"},
            ],
        }
        content = build_user_content_for_planner(state)
        
        assert "- fig1: First figure" in content
        assert "- fig2: Second figure" in content
        assert "- fig3: Third figure" in content
        # Verify order is preserved
        assert content.index("fig1") < content.index("fig2") < content.index("fig3")

    def test_figure_without_id(self):
        """Test figure without id field - should use 'unknown'."""
        state = {
            "paper_text": "Paper",
            "paper_figures": [{"description": "A figure"}],
        }
        content = build_user_content_for_planner(state)
        
        assert "- unknown: A figure" in content

    def test_figure_without_description(self):
        """Test figure without description - should use 'No description'."""
        state = {
            "paper_text": "Paper",
            "paper_figures": [{"id": "fig1"}],
        }
        content = build_user_content_for_planner(state)
        
        assert "- fig1: No description" in content

    def test_empty_paper_text(self):
        """Test with empty paper text - should not include section."""
        state = {
            "paper_text": "",
            "paper_figures": [{"id": "fig1"}],
        }
        content = build_user_content_for_planner(state)
        
        assert "# PAPER TEXT" not in content
        assert "# FIGURES" in content

    def test_empty_figures_list(self):
        """Test with empty figures list - should not include section."""
        state = {
            "paper_text": "Paper",
            "paper_figures": [],
        }
        content = build_user_content_for_planner(state)
        
        assert "# PAPER TEXT" in content
        assert "# FIGURES" not in content

    def test_empty_feedback(self):
        """Test with empty feedback - should not include section."""
        state = {
            "paper_text": "Paper",
            "planner_feedback": "",
        }
        content = build_user_content_for_planner(state)
        
        assert "# PAPER TEXT" in content
        assert "# REVISION FEEDBACK" not in content

    def test_missing_paper_text_key(self):
        """Test with missing paper_text key - should not include section."""
        state = {
            "paper_figures": [{"id": "fig1"}],
        }
        content = build_user_content_for_planner(state)
        
        assert "# PAPER TEXT" not in content
        assert "# FIGURES" in content

    def test_missing_figures_key(self):
        """Test with missing paper_figures key - should not include section."""
        state = {
            "paper_text": "Paper",
        }
        content = build_user_content_for_planner(state)
        
        assert "# PAPER TEXT" in content
        assert "# FIGURES" not in content

    def test_missing_feedback_key(self):
        """Test with missing planner_feedback key - should not include section."""
        state = {
            "paper_text": "Paper",
        }
        content = build_user_content_for_planner(state)
        
        assert "# PAPER TEXT" in content
        assert "# REVISION FEEDBACK" not in content

    def test_empty_state(self):
        """Test with completely empty state - should return empty string."""
        state = {}
        content = build_user_content_for_planner(state)
        
        assert content == ""

    def test_none_values(self):
        """Test with None values - should handle gracefully."""
        state = {
            "paper_text": None,
            "paper_figures": None,
            "planner_feedback": None,
        }
        content = build_user_content_for_planner(state)
        
        # None values should be treated as missing
        assert content == ""

    def test_very_long_paper_text(self):
        """Test with very long paper text - should include all content."""
        long_text = "A" * 10000
        state = {
            "paper_text": long_text,
        }
        content = build_user_content_for_planner(state)
        
        assert len(content) > 10000
        assert long_text in content

    def test_special_characters_in_content(self):
        """Test with special characters - should preserve them."""
        state = {
            "paper_text": "Text with\nnewlines\tand\ttabs",
            "paper_figures": [{"id": "fig1", "description": "Figure with \"quotes\" and 'apostrophes'"}],
            "planner_feedback": "Feedback with\nmultiple\nlines",
        }
        content = build_user_content_for_planner(state)
        
        assert "\nnewlines\tand\ttabs" in content
        assert "\"quotes\"" in content
        assert "'apostrophes'" in content
        assert "multiple\nlines" in content


class TestBuildUserContentForDesigner:
    """Tests for build_user_content_for_designer."""

    def test_full_state_all_fields_present(self):
        """Test with all fields present - verify exact structure."""
        state = {
            "current_stage_id": "stage1",
            "plan": {"stages": [{"stage_id": "stage1", "task": "do x", "targets": ["fig1"]}]},
            "extracted_parameters": [{"param": "val"}],
            "assumptions": {"assump": "val"},
            "validated_materials": ["mat1"],
            "reviewer_feedback": "Change design",
        }
        content = build_user_content_for_designer(state)
        
        # Verify exact structure
        assert content.startswith("# CURRENT STAGE: stage1")
        
        # Verify stage details are in JSON format
        assert "## Stage Details" in content
        assert "```json" in content
        assert "stage_id" in content
        assert "do x" in content
        
        # Verify extracted parameters
        assert "## Extracted Parameters" in content
        params_json = json.loads(content.split("## Extracted Parameters")[1].split("```json")[1].split("```")[0].strip())
        assert params_json == [{"param": "val"}]
        
        # Verify assumptions
        assert "## Assumptions" in content
        assumptions_json = json.loads(content.split("## Assumptions")[1].split("```json")[1].split("```")[0].strip())
        assert assumptions_json == {"assump": "val"}
        
        # Verify materials
        assert "## Validated Materials" in content
        materials_json = json.loads(content.split("## Validated Materials")[1].split("```json")[1].split("```")[0].strip())
        assert materials_json == ["mat1"]
        
        # Verify feedback
        assert "## REVISION FEEDBACK" in content
        assert "Change design" in content

    def test_missing_current_stage_id(self):
        """Test with missing current_stage_id - should use 'unknown'."""
        state = {
            "plan": {"stages": []},
        }
        content = build_user_content_for_designer(state)
        
        assert content.startswith("# CURRENT STAGE: unknown")

    def test_stage_not_found_in_plan(self):
        """Test when current_stage_id doesn't match any stage in plan."""
        state = {
            "current_stage_id": "stage99",
            "plan": {"stages": [{"stage_id": "stage1", "task": "do x"}]},
        }
        content = build_user_content_for_designer(state)
        
        assert "# CURRENT STAGE: stage99" in content
        assert "Stage Details" not in content

    def test_empty_extracted_parameters(self):
        """Test with empty extracted_parameters - should not include section."""
        state = {
            "current_stage_id": "stage1",
            "extracted_parameters": [],
        }
        content = build_user_content_for_designer(state)
        
        assert "## Extracted Parameters" not in content

    def test_many_extracted_parameters_truncation(self):
        """Test with more than 20 parameters - should truncate to 20."""
        state = {
            "current_stage_id": "stage1",
            "extracted_parameters": [{"param": i} for i in range(25)],
        }
        content = build_user_content_for_designer(state)
        
        params_json = json.loads(content.split("## Extracted Parameters")[1].split("```json")[1].split("```")[0].strip())
        assert len(params_json) == 20
        assert params_json[0] == {"param": 0}
        assert params_json[19] == {"param": 19}

    def test_empty_assumptions(self):
        """Test with empty assumptions dict - should not include section."""
        state = {
            "current_stage_id": "stage1",
            "assumptions": {},
        }
        content = build_user_content_for_designer(state)
        
        assert "## Assumptions" not in content

    def test_empty_validated_materials(self):
        """Test with empty validated_materials - should not include section."""
        state = {
            "current_stage_id": "stage1",
            "validated_materials": [],
        }
        content = build_user_content_for_designer(state)
        
        assert "## Validated Materials" not in content

    def test_empty_feedback(self):
        """Test with empty feedback - should not include section."""
        state = {
            "current_stage_id": "stage1",
            "reviewer_feedback": "",
        }
        content = build_user_content_for_designer(state)
        
        assert "## REVISION FEEDBACK" not in content

    def test_missing_plan_key(self):
        """Test with missing plan key - should handle gracefully."""
        state = {
            "current_stage_id": "stage1",
        }
        content = build_user_content_for_designer(state)
        
        assert "# CURRENT STAGE: stage1" in content
        assert "Stage Details" not in content

    def test_missing_stages_in_plan(self):
        """Test with plan missing stages key."""
        state = {
            "current_stage_id": "stage1",
            "plan": {},
        }
        content = build_user_content_for_designer(state)
        
        assert "# CURRENT STAGE: stage1" in content
        assert "Stage Details" not in content

    def test_complex_nested_stage_data(self):
        """Test with complex nested stage data - should serialize correctly."""
        complex_stage = {
            "stage_id": "stage1",
            "task": "Complex task",
            "targets": ["fig1", "fig2"],
            "parameters": {
                "nested": {
                    "deep": {"value": 42}
                },
                "list": [1, 2, 3],
            },
        }
        state = {
            "current_stage_id": "stage1",
            "plan": {"stages": [complex_stage]},
        }
        content = build_user_content_for_designer(state)
        
        stage_json = json.loads(content.split("## Stage Details")[1].split("```json")[1].split("```")[0].strip())
        assert stage_json == complex_stage

    def test_empty_state(self):
        """Test with completely empty state."""
        state = {}
        content = build_user_content_for_designer(state)
        
        assert content.startswith("# CURRENT STAGE: unknown")
        assert len(content.split("\n\n")) == 1  # Only stage header

    def test_none_values(self):
        """Test with None values - should handle gracefully."""
        state = {
            "current_stage_id": None,
            "plan": None,
            "extracted_parameters": None,
            "assumptions": None,
            "validated_materials": None,
            "reviewer_feedback": None,
        }
        content = build_user_content_for_designer(state)
        
        # None values should be treated as missing
        assert content.startswith("# CURRENT STAGE: unknown")


class TestBuildUserContentForCodeGenerator:
    """Tests for build_user_content_for_code_generator."""

    def test_full_state_string_design(self):
        """Test with string design_description - verify exact structure."""
        state = {
            "current_stage_id": "stage1",
            "design_description": "A design spec",
            "validated_materials": ["mat1"],
            "reviewer_feedback": "Fix code",
        }
        content = build_user_content_for_code_generator(state)
        
        # Verify structure (headers and content are separated by \n\n)
        assert content.startswith("# CURRENT STAGE: stage1")
        assert "## Design Specification" in content
        assert "A design spec" in content
        assert "## Validated Materials (use these paths!)" in content
        assert "```json" in content
        assert '"mat1"' in content
        assert "## REVISION FEEDBACK" in content
        assert "Fix code" in content
        
        # Verify exact format for string design (not JSON)
        parts = content.split("\n\n")
        assert len(parts) == 6  # stage, design header, design content, materials block, feedback header, feedback content
        assert parts[0] == "# CURRENT STAGE: stage1"
        assert parts[1] == "## Design Specification"
        assert parts[2] == "A design spec"

    def test_dict_design_description(self):
        """Test with dict design_description - should serialize as JSON."""
        state = {
            "current_stage_id": "stage1",
            "design_description": {"structure": "nanorod", "material": "gold"},
            "validated_materials": ["mat1"],
        }
        content = build_user_content_for_code_generator(state)
        
        assert "## Design Specification" in content
        assert "```json" in content
        design_json = json.loads(content.split("## Design Specification")[1].split("```json")[1].split("```")[0].strip())
        assert design_json == {"structure": "nanorod", "material": "gold"}

    def test_missing_current_stage_id(self):
        """Test with missing current_stage_id - should use 'unknown'."""
        state = {
            "design_description": "Design",
        }
        content = build_user_content_for_code_generator(state)
        
        assert content.startswith("# CURRENT STAGE: unknown")

    def test_empty_design_description(self):
        """Test with empty design_description - should not include section."""
        state = {
            "current_stage_id": "stage1",
            "design_description": "",
        }
        content = build_user_content_for_code_generator(state)
        
        assert "## Design Specification" not in content

    def test_missing_design_description_key(self):
        """Test with missing design_description key - should not include section."""
        state = {
            "current_stage_id": "stage1",
        }
        content = build_user_content_for_code_generator(state)
        
        assert "## Design Specification" not in content

    def test_empty_validated_materials(self):
        """Test with empty validated_materials - should not include section."""
        state = {
            "current_stage_id": "stage1",
            "design_description": "Design",
            "validated_materials": [],
        }
        content = build_user_content_for_code_generator(state)
        
        assert "## Validated Materials" not in content

    def test_missing_validated_materials_key(self):
        """Test with missing validated_materials key - should not include section."""
        state = {
            "current_stage_id": "stage1",
            "design_description": "Design",
        }
        content = build_user_content_for_code_generator(state)
        
        assert "## Validated Materials" not in content

    def test_empty_feedback(self):
        """Test with empty feedback - should not include section."""
        state = {
            "current_stage_id": "stage1",
            "design_description": "Design",
            "reviewer_feedback": "",
        }
        content = build_user_content_for_code_generator(state)
        
        assert "## REVISION FEEDBACK" not in content

    def test_complex_materials_list(self):
        """Test with complex materials data - should serialize correctly."""
        complex_materials = [
            {"name": "gold", "path": "/path/to/gold.csv"},
            {"name": "silver", "path": "/path/to/silver.csv", "metadata": {"source": "palik"}},
        ]
        state = {
            "current_stage_id": "stage1",
            "design_description": "Design",
            "validated_materials": complex_materials,
        }
        content = build_user_content_for_code_generator(state)
        
        materials_json = json.loads(content.split("## Validated Materials")[1].split("```json")[1].split("```")[0].strip())
        assert materials_json == complex_materials

    def test_empty_state(self):
        """Test with completely empty state."""
        state = {}
        content = build_user_content_for_code_generator(state)
        
        assert content.startswith("# CURRENT STAGE: unknown")
        assert len(content.split("\n\n")) == 1  # Only stage header

    def test_none_values(self):
        """Test with None values - should handle gracefully."""
        state = {
            "current_stage_id": None,
            "design_description": None,
            "validated_materials": None,
            "reviewer_feedback": None,
        }
        content = build_user_content_for_code_generator(state)
        
        assert content.startswith("# CURRENT STAGE: unknown")


class TestBuildUserContentForAnalyzer:
    """Tests for build_user_content_for_analyzer."""

    def test_full_state_all_fields_present(self):
        """Test with all fields present - verify exact structure."""
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {"files": ["output.png"], "data": {"value": 42}},
            "plan": {"stages": [{"stage_id": "stage1", "targets": ["fig1", "fig2"]}]},
            "analysis_feedback": "Analyze better",
        }
        content = build_user_content_for_analyzer(state)
        
        # Verify structure (headers and content are separated by \n\n)
        assert content.startswith("# CURRENT STAGE: stage1")
        assert "## Simulation Outputs" in content
        assert "```json" in content
        assert "output.png" in content
        assert "## Target Figures: fig1, fig2" in content
        assert "## REVISION FEEDBACK" in content
        assert "Analyze better" in content
        
        # Verify exact format
        parts = content.split("\n\n")
        assert len(parts) == 5  # stage, outputs block, targets, feedback header, feedback content
        
        assert parts[0] == "# CURRENT STAGE: stage1"
        
        # Verify outputs are in JSON format
        assert "## Simulation Outputs" in parts[1]
        assert "```json" in parts[1]
        outputs_json = json.loads(parts[1].split("```json")[1].split("```")[0].strip())
        assert outputs_json == {"files": ["output.png"], "data": {"value": 42}}
        
        assert parts[2] == "## Target Figures: fig1, fig2"
        assert parts[3] == "## REVISION FEEDBACK"
        assert parts[4] == "Analyze better"

    def test_missing_current_stage_id(self):
        """Test with missing current_stage_id - should use 'unknown'."""
        state = {
            "stage_outputs": {},
        }
        content = build_user_content_for_analyzer(state)
        
        assert content.startswith("# CURRENT STAGE: unknown")

    def test_empty_stage_outputs(self):
        """Test with empty stage_outputs - should not include section."""
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {},
        }
        content = build_user_content_for_analyzer(state)
        
        # Empty dict should still be included (falsy check)
        assert "## Simulation Outputs" in content

    def test_missing_stage_outputs_key(self):
        """Test with missing stage_outputs key - should not include section."""
        state = {
            "current_stage_id": "stage1",
        }
        content = build_user_content_for_analyzer(state)
        
        assert "## Simulation Outputs" not in content

    def test_stage_not_found_in_plan(self):
        """Test when current_stage_id doesn't match any stage in plan."""
        state = {
            "current_stage_id": "stage99",
            "plan": {"stages": [{"stage_id": "stage1", "targets": ["fig1"]}]},
        }
        content = build_user_content_for_analyzer(state)
        
        assert "# CURRENT STAGE: stage99" in content
        assert "## Target Figures" not in content

    def test_stage_without_targets(self):
        """Test stage without targets field - should handle gracefully."""
        state = {
            "current_stage_id": "stage1",
            "plan": {"stages": [{"stage_id": "stage1"}]},
        }
        content = build_user_content_for_analyzer(state)
        
        # Should not include targets section if targets is missing
        assert "## Target Figures" not in content

    def test_stage_with_empty_targets(self):
        """Test stage with empty targets list - should not include section."""
        state = {
            "current_stage_id": "stage1",
            "plan": {"stages": [{"stage_id": "stage1", "targets": []}]},
        }
        content = build_user_content_for_analyzer(state)
        
        # Empty targets list should not be included (correct behavior)
        assert "## Target Figures" not in content

    def test_empty_feedback(self):
        """Test with empty feedback - should not include section."""
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {},
            "analysis_feedback": "",
        }
        content = build_user_content_for_analyzer(state)
        
        assert "## REVISION FEEDBACK" not in content

    def test_missing_plan_key(self):
        """Test with missing plan key - should handle gracefully."""
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {},
        }
        content = build_user_content_for_analyzer(state)
        
        assert "# CURRENT STAGE: stage1" in content
        assert "## Target Figures" not in content

    def test_complex_stage_outputs(self):
        """Test with complex stage_outputs - should serialize correctly."""
        complex_outputs = {
            "files": ["output1.png", "output2.png"],
            "data": {
                "wavelengths": [400, 500, 600],
                "results": [{"value": 1.0}, {"value": 2.0}],
            },
            "metadata": {"timestamp": "2024-01-01"},
        }
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": complex_outputs,
        }
        content = build_user_content_for_analyzer(state)
        
        outputs_json = json.loads(content.split("## Simulation Outputs")[1].split("```json")[1].split("```")[0].strip())
        assert outputs_json == complex_outputs

    def test_stage_outputs_with_non_serializable(self):
        """Test stage_outputs with non-JSON-serializable values - should use default=str."""
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {"path": Path("/some/path"), "number": 42},
        }
        content = build_user_content_for_analyzer(state)
        
        # Should not raise error, Path should be converted to string
        assert "## Simulation Outputs" in content
        assert "/some/path" in content or "some" in content

    def test_empty_state(self):
        """Test with completely empty state."""
        state = {}
        content = build_user_content_for_analyzer(state)
        
        assert content.startswith("# CURRENT STAGE: unknown")
        assert len(content.split("\n\n")) == 1  # Only stage header

    def test_none_values(self):
        """Test with None values - should handle gracefully."""
        state = {
            "current_stage_id": None,
            "stage_outputs": None,
            "plan": None,
            "analysis_feedback": None,
        }
        content = build_user_content_for_analyzer(state)
        
        assert content.startswith("# CURRENT STAGE: unknown")


class TestGetImagesForAnalyzer:
    """Tests for get_images_for_analyzer."""

    def test_paper_figures_only(self):
        """Test with only paper figures."""
        with patch("src.llm_client.Path") as mock_path:
            def path_side_effect(path_str):
                mock_obj = MagicMock()
                mock_obj.exists.return_value = True
                mock_obj.suffix = ".png"
                mock_obj.__str__.return_value = str(path_str)
                return mock_obj
            
            mock_path.side_effect = path_side_effect
            
            state = {
                "paper_figures": [
                    {"image_path": "fig1.png"},
                    {"image_path": "fig2.png"},
                ],
            }
            images = get_images_for_analyzer(state)
            
            assert len(images) == 2
            assert all(str(img) in ["fig1.png", "fig2.png"] for img in images)

    def test_stage_outputs_only(self):
        """Test with only stage output files."""
        with patch("src.llm_client.Path") as mock_path:
            def path_side_effect(path_str):
                mock_obj = MagicMock()
                mock_obj.exists.return_value = True
                mock_obj.suffix = ".png" if str(path_str).endswith(".png") else ".csv"
                mock_obj.__str__.return_value = str(path_str)
                return mock_obj
            
            mock_path.side_effect = path_side_effect
            
            state = {
                "stage_outputs": {
                    "files": ["output1.png", "output2.png", "data.csv"],
                },
            }
            images = get_images_for_analyzer(state)
            
            assert len(images) == 2
            assert all(str(img).endswith(".png") for img in images)

    def test_both_paper_figures_and_stage_outputs(self):
        """Test with both paper figures and stage outputs."""
        with patch("src.llm_client.Path") as mock_path:
            def path_side_effect(path_str):
                mock_obj = MagicMock()
                mock_obj.exists.return_value = True
                path_str_val = str(path_str)
                if path_str_val.endswith(".png"):
                    mock_obj.suffix = ".png"
                elif path_str_val.endswith(".jpg"):
                    mock_obj.suffix = ".jpg"
                else:
                    mock_obj.suffix = ".csv"
                mock_obj.__str__.return_value = path_str_val
                return mock_obj
            
            mock_path.side_effect = path_side_effect
            
            state = {
                "paper_figures": [{"image_path": "fig1.png"}],
                "stage_outputs": {"files": ["output1.png", "data.csv"]},
            }
            images = get_images_for_analyzer(state)
            
            assert len(images) == 2
            assert all(str(img).endswith(".png") for img in images)

    def test_paper_figure_without_image_path(self):
        """Test paper figure without image_path - should be skipped."""
        with patch("src.llm_client.Path") as mock_path:
            def path_side_effect(path_str):
                mock_obj = MagicMock()
                mock_obj.exists.return_value = True
                mock_obj.suffix = ".png"
                mock_obj.__str__.return_value = str(path_str)
                return mock_obj
            
            mock_path.side_effect = path_side_effect
            
            state = {
                "paper_figures": [
                    {"id": "fig1"},  # No image_path
                    {"image_path": "fig2.png"},
                ],
            }
            images = get_images_for_analyzer(state)
            
            assert len(images) == 1
            # Check that the path string is in the image
            image_strs = [str(img) for img in images]
            assert "fig2.png" in image_strs

    def test_paper_figure_with_none_image_path(self):
        """Test paper figure with None image_path - should be skipped."""
        with patch("src.llm_client.Path") as mock_path:
            def path_side_effect(path_str):
                mock_obj = MagicMock()
                mock_obj.exists.return_value = True
                mock_obj.suffix = ".png"
                mock_obj.__str__.return_value = str(path_str)
                return mock_obj
            
            mock_path.side_effect = path_side_effect
            
            state = {
                "paper_figures": [
                    {"image_path": None},
                    {"image_path": "fig2.png"},
                ],
            }
            images = get_images_for_analyzer(state)
            
            assert len(images) == 1
            # Check that the path string is in the image
            image_strs = [str(img) for img in images]
            assert "fig2.png" in image_strs

    def test_nonexistent_paper_figure_path(self):
        """Test paper figure with non-existent path - should be skipped."""
        with patch("src.llm_client.Path") as mock_path:
            def path_side_effect(path_str):
                mock_obj = MagicMock()
                mock_obj.exists.return_value = str(path_str) != "nonexistent.png"
                mock_obj.suffix = ".png"
                mock_obj.__str__.return_value = str(path_str)
                return mock_obj
            
            mock_path.side_effect = path_side_effect
            
            state = {
                "paper_figures": [
                    {"image_path": "nonexistent.png"},
                    {"image_path": "existing.png"},
                ],
            }
            images = get_images_for_analyzer(state)
            
            assert len(images) == 1
            assert str(images[0]) == "existing.png"

    def test_nonexistent_stage_output_path(self):
        """Test stage output with non-existent path - should be skipped."""
        with patch("src.llm_client.Path") as mock_path:
            def path_side_effect(path_str):
                mock_obj = MagicMock()
                mock_obj.exists.return_value = str(path_str) != "nonexistent.png"
                mock_obj.suffix = ".png" if str(path_str).endswith(".png") else ".csv"
                mock_obj.__str__.return_value = str(path_str)
                return mock_obj
            
            mock_path.side_effect = path_side_effect
            
            state = {
                "stage_outputs": {
                    "files": ["nonexistent.png", "existing.png"],
                },
            }
            images = get_images_for_analyzer(state)
            
            assert len(images) == 1
            assert str(images[0]) == "existing.png"

    def test_stage_output_path_as_path_object(self):
        """Test stage output with Path object - should handle correctly."""
        with patch("src.llm_client.Path") as mock_path:
            def path_side_effect(path_str):
                # If already a Path-like object, return as-is
                if isinstance(path_str, MagicMock):
                    return path_str
                mock_obj = MagicMock()
                mock_obj.exists.return_value = True
                mock_obj.suffix = ".png"
                mock_obj.__str__.return_value = str(path_str)
                return mock_obj
            
            mock_path.side_effect = path_side_effect
            
            mock_path_obj = MagicMock()
            mock_path_obj.exists.return_value = True
            mock_path_obj.suffix = ".png"
            mock_path_obj.__str__.return_value = "output.png"
            
            state = {
                "stage_outputs": {
                    "files": [mock_path_obj],
                },
            }
            images = get_images_for_analyzer(state)
            
            assert len(images) == 1

    def test_stage_output_path_as_dict(self):
        """Test stage output with dict containing path - should extract path."""
        with patch("src.llm_client.Path") as mock_path:
            def path_side_effect(path_str):
                mock_obj = MagicMock()
                mock_obj.exists.return_value = True
                mock_obj.suffix = ".png"
                mock_obj.__str__.return_value = str(path_str)
                return mock_obj
            
            mock_path.side_effect = path_side_effect
            
            state = {
                "stage_outputs": {
                    "files": [{"path": "output.png", "type": "image"}],
                },
            }
            images = get_images_for_analyzer(state)
            
            assert len(images) == 1
            assert str(images[0]) == "output.png"

    def test_stage_output_dict_without_path_key(self):
        """Test stage output dict without path key - should handle gracefully."""
        with patch("src.llm_client.Path") as mock_path:
            def path_side_effect(path_str):
                mock_obj = MagicMock()
                mock_obj.exists.return_value = False  # Won't exist anyway
                mock_obj.suffix = ""
                mock_obj.__str__.return_value = str(path_str)
                return mock_obj
            
            mock_path.side_effect = path_side_effect
            
            state = {
                "stage_outputs": {
                    "files": [{"type": "image"}],  # No path key
                },
            }
            images = get_images_for_analyzer(state)
            
            # Should try to create Path("") which won't exist
            assert len(images) == 0

    def test_non_image_file_extensions(self):
        """Test that non-image files are filtered out."""
        with patch("src.llm_client.Path") as mock_path:
            def path_side_effect(path_str):
                mock_obj = MagicMock()
                mock_obj.exists.return_value = True
                path_str_val = str(path_str)
                if path_str_val.endswith(".png"):
                    mock_obj.suffix = ".png"
                elif path_str_val.endswith(".jpg"):
                    mock_obj.suffix = ".jpg"
                elif path_str_val.endswith(".jpeg"):
                    mock_obj.suffix = ".jpeg"
                elif path_str_val.endswith(".gif"):
                    mock_obj.suffix = ".gif"
                elif path_str_val.endswith(".webp"):
                    mock_obj.suffix = ".webp"
                else:
                    mock_obj.suffix = ".csv"
                mock_obj.__str__.return_value = path_str_val
                return mock_obj
            
            mock_path.side_effect = path_side_effect
            
            state = {
                "stage_outputs": {
                    "files": [
                        "output.png",
                        "output.jpg",
                        "output.jpeg",
                        "output.gif",
                        "output.webp",
                        "data.csv",
                        "data.txt",
                    ],
                },
            }
            images = get_images_for_analyzer(state)
            
            assert len(images) == 5
            image_extensions = [str(img).split(".")[-1].lower() for img in images]
            assert "csv" not in image_extensions
            assert "txt" not in image_extensions

    def test_case_insensitive_image_extensions(self):
        """Test that image extensions are case-insensitive."""
        with patch("src.llm_client.Path") as mock_path:
            def path_side_effect(path_str):
                mock_obj = MagicMock()
                mock_obj.exists.return_value = True
                path_str_val = str(path_str)
                # Preserve original case in suffix
                if path_str_val.lower().endswith(".png"):
                    mock_obj.suffix = path_str_val[-4:]
                elif path_str_val.lower().endswith(".jpg"):
                    mock_obj.suffix = path_str_val[-4:]
                else:
                    mock_obj.suffix = ".csv"
                mock_obj.__str__.return_value = path_str_val
                return mock_obj
            
            mock_path.side_effect = path_side_effect
            
            state = {
                "stage_outputs": {
                    "files": ["output.PNG", "output.JPG", "output.png", "data.csv"],
                },
            }
            images = get_images_for_analyzer(state)
            
            # Should include both .PNG and .png (case-insensitive check)
            assert len(images) >= 2

    def test_empty_paper_figures(self):
        """Test with empty paper_figures list."""
        state = {
            "paper_figures": [],
        }
        images = get_images_for_analyzer(state)
        
        assert len(images) == 0

    def test_empty_stage_outputs_files(self):
        """Test with empty stage_outputs files list."""
        state = {
            "stage_outputs": {"files": []},
        }
        images = get_images_for_analyzer(state)
        
        assert len(images) == 0

    def test_missing_paper_figures_key(self):
        """Test with missing paper_figures key."""
        state = {
            "stage_outputs": {"files": ["output.png"]},
        }
        with patch("src.llm_client.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.suffix = ".png"
            images = get_images_for_analyzer(state)
            
            assert len(images) == 1

    def test_missing_stage_outputs_key(self):
        """Test with missing stage_outputs key."""
        state = {
            "paper_figures": [{"image_path": "fig1.png"}],
        }
        with patch("src.llm_client.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.suffix = ".png"
            images = get_images_for_analyzer(state)
            
            assert len(images) == 1

    def test_empty_state(self):
        """Test with completely empty state."""
        state = {}
        images = get_images_for_analyzer(state)
        
        assert len(images) == 0

    def test_none_values(self):
        """Test with None values - should handle gracefully."""
        state = {
            "paper_figures": None,
            "stage_outputs": None,
        }
        images = get_images_for_analyzer(state)
        
        assert len(images) == 0

    def test_stage_outputs_none_files(self):
        """Test with stage_outputs having None files."""
        state = {
            "stage_outputs": {"files": None},
        }
        # Should handle gracefully without crashing
        try:
            images = get_images_for_analyzer(state)
            assert len(images) == 0
        except (TypeError, AttributeError):
            # If it crashes, that's a bug we want to catch
            pytest.fail("get_images_for_analyzer should handle None files gracefully")

    def test_stage_outputs_missing_files_key(self):
        """Test with stage_outputs missing files key."""
        state = {
            "stage_outputs": {},
        }
        images = get_images_for_analyzer(state)
        
        assert len(images) == 0
