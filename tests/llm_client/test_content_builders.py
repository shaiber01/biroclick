"""Content builder tests for `src.llm_client`."""

from unittest.mock import MagicMock, patch

from src.llm_client import (
    build_user_content_for_analyzer,
    build_user_content_for_code_generator,
    build_user_content_for_designer,
    build_user_content_for_planner,
    get_images_for_analyzer,
)


class TestContentBuilders:
    """Tests for user content builder functions."""

    def test_build_user_content_for_planner(self):
        state = {
            "paper_text": "Full paper content",
            "paper_figures": [{"id": "fig1", "description": "A figure"}],
            "planner_feedback": "Please revise",
        }
        content = build_user_content_for_planner(state)
        assert "# PAPER TEXT" in content
        assert "Full paper content" in content
        assert "# FIGURES" in content
        assert "fig1: A figure" in content
        assert "# REVISION FEEDBACK" in content
        assert "Please revise" in content

    def test_build_user_content_for_designer(self):
        state = {
            "current_stage_id": "stage1",
            "plan": {"stages": [{"stage_id": "stage1", "task": "do x"}]},
            "extracted_parameters": [{"param": "val"}],
            "assumptions": {"assump": "val"},
            "validated_materials": ["mat1"],
            "reviewer_feedback": "Change design",
        }
        content = build_user_content_for_designer(state)
        assert "# CURRENT STAGE: stage1" in content
        assert "Stage Details" in content
        assert "do x" in content
        assert "Extracted Parameters" in content
        assert "Assumptions" in content
        assert "Validated Materials" in content
        assert "# REVISION FEEDBACK" in content
        assert "Change design" in content

    def test_build_user_content_for_code_generator(self):
        state = {
            "current_stage_id": "stage1",
            "design_description": "A design spec",
            "validated_materials": ["mat1"],
            "reviewer_feedback": "Fix code",
        }
        content = build_user_content_for_code_generator(state)
        assert "# CURRENT STAGE: stage1" in content
        assert "Design Specification" in content
        assert "A design spec" in content
        assert "Validated Materials" in content
        assert "# REVISION FEEDBACK" in content
        assert "Fix code" in content

    def test_build_user_content_for_analyzer(self):
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {"files": ["output.png"]},
            "plan": {"stages": [{"stage_id": "stage1", "targets": ["fig1"]}]},
            "analysis_feedback": "Analyze better",
        }
        content = build_user_content_for_analyzer(state)
        assert "# CURRENT STAGE: stage1" in content
        assert "Simulation Outputs" in content
        assert "output.png" in content
        assert "Target Figures: fig1" in content
        assert "# REVISION FEEDBACK" in content
        assert "Analyze better" in content

    @patch("src.llm_client.Path")
    def test_get_images_for_analyzer(self, mock_path):
        mock_path.return_value.exists.return_value = True

        state = {
            "paper_figures": [{"image_path": "fig1.png"}],
            "stage_outputs": {"files": ["output1.png", "data.csv"]},
        }

        def path_side_effect(path_str):
            mock_obj = MagicMock()
            mock_obj.exists.return_value = True
            mock_obj.suffix = ".png" if str(path_str).endswith(".png") else ".csv"
            mock_obj.__str__.return_value = str(path_str)
            return mock_obj

        mock_path.side_effect = path_side_effect

        images = get_images_for_analyzer(state)

        assert len(images) == 2
        assert all(str(img).endswith(".png") for img in images)


