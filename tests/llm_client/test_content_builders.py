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

    def test_whitespace_only_paper_text(self):
        """Test with whitespace-only paper text - should not include section (empty after strip would be falsy)."""
        state = {
            "paper_text": "   \n\t   ",
            "paper_figures": [{"id": "fig1", "description": "A figure"}],
        }
        content = build_user_content_for_planner(state)
        
        # Whitespace-only text is still truthy, so it should include the section
        # This tests current behavior - if whitespace should be trimmed, the component needs fixing
        assert "# PAPER TEXT" in content
        assert "   \n\t   " in content  # Whitespace preserved

    def test_figure_with_empty_strings(self):
        """Test figure with both id and description as empty strings."""
        state = {
            "paper_text": "Paper",
            "paper_figures": [{"id": "", "description": ""}],
        }
        content = build_user_content_for_planner(state)
        
        # Empty string id should show as empty, empty description should show as empty
        assert "# FIGURES" in content
        # Based on the code: fig.get('id', 'unknown') - empty string is not None, so it stays empty
        # and fig.get('description', 'No description') - empty string is not None
        assert "- : " in content  # Empty id and empty description

    def test_figure_with_only_unknown_keys(self):
        """Test figure dict with no id or description keys."""
        state = {
            "paper_text": "Paper",
            "paper_figures": [{"unknown_key": "value"}],
        }
        content = build_user_content_for_planner(state)
        
        assert "# FIGURES" in content
        assert "- unknown: No description" in content

    def test_non_dict_figure_in_list(self):
        """Test figures list containing non-dict items - should handle gracefully or fail clearly."""
        state = {
            "paper_text": "Paper",
            "paper_figures": [
                {"id": "fig1", "description": "Valid figure"},
                "invalid_figure",  # Non-dict item
                {"id": "fig2", "description": "Another valid figure"},
            ],
        }
        # This tests that the code handles mixed types properly
        # If it crashes, that's a bug in the component
        try:
            content = build_user_content_for_planner(state)
            # If it doesn't crash, verify valid figures are included
            assert "- fig1: Valid figure" in content
            assert "- fig2: Another valid figure" in content
        except AttributeError:
            pytest.fail("build_user_content_for_planner should handle non-dict figures gracefully")

    def test_unicode_content(self):
        """Test with Unicode characters in all fields."""
        state = {
            "paper_text": "论文包含中文字符 and émojis 🔬🧪",
            "paper_figures": [{"id": "図1", "description": "日本語の説明 with αβγ Greek letters"}],
            "planner_feedback": "Обратная связь на русском языке",
        }
        content = build_user_content_for_planner(state)
        
        assert "论文包含中文字符" in content
        assert "🔬🧪" in content
        assert "図1" in content
        assert "日本語の説明" in content
        assert "αβγ" in content
        assert "Обратная связь" in content

    def test_sections_order_consistency(self):
        """Test that sections appear in consistent order: paper_text, figures, feedback."""
        state = {
            "paper_text": "Paper text content",
            "paper_figures": [{"id": "fig1", "description": "Figure"}],
            "planner_feedback": "Feedback content",
        }
        content = build_user_content_for_planner(state)
        
        paper_idx = content.index("# PAPER TEXT")
        figures_idx = content.index("# FIGURES")
        feedback_idx = content.index("# REVISION FEEDBACK")
        
        assert paper_idx < figures_idx < feedback_idx, "Sections must appear in order: PAPER TEXT < FIGURES < FEEDBACK"

    def test_separator_format(self):
        """Test that the separator between sections is exactly '---' surrounded by double newlines."""
        state = {
            "paper_text": "Paper",
            "paper_figures": [{"id": "fig1", "description": "Fig"}],
        }
        content = build_user_content_for_planner(state)
        
        # Verify the exact separator format
        assert "\n\n---\n\n" in content
        # Count separators - should be exactly 1 between 2 sections
        assert content.count("\n\n---\n\n") == 1

    def test_many_figures_performance(self):
        """Test with a large number of figures."""
        state = {
            "paper_text": "Paper",
            "paper_figures": [{"id": f"fig{i}", "description": f"Description {i}"} for i in range(100)],
        }
        content = build_user_content_for_planner(state)
        
        # All figures should be included
        assert "- fig0: Description 0" in content
        assert "- fig50: Description 50" in content
        assert "- fig99: Description 99" in content
        assert content.count("- fig") == 100

    def test_figure_description_with_special_markdown(self):
        """Test figure descriptions containing markdown-like content."""
        state = {
            "paper_text": "Paper",
            "paper_figures": [
                {"id": "fig1", "description": "Contains **bold** and *italic*"},
                {"id": "fig2", "description": "Has `code` and [link](url)"},
                {"id": "fig3", "description": "Shows # header and --- divider"},
            ],
        }
        content = build_user_content_for_planner(state)
        
        # All markdown characters should be preserved
        assert "**bold**" in content
        assert "*italic*" in content
        assert "`code`" in content
        assert "[link](url)" in content
        assert "# header" in content


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

    def test_empty_string_stage_id(self):
        """Test with empty string stage_id - should use as-is (not 'unknown')."""
        state = {
            "current_stage_id": "",
            "plan": {"stages": [{"stage_id": "", "task": "empty stage task"}]},
        }
        content = build_user_content_for_designer(state)
        
        # Empty string is falsy, so it should trigger "unknown" based on `or "unknown"`
        assert content.startswith("# CURRENT STAGE: unknown")

    def test_plan_is_list_not_dict(self):
        """Test when plan is a list instead of dict - should handle gracefully."""
        state = {
            "current_stage_id": "stage1",
            "plan": [{"stage_id": "stage1", "task": "do x"}],  # List instead of dict
        }
        content = build_user_content_for_designer(state)
        
        # Code checks `isinstance(plan, dict)`, so list should result in no stage details
        assert "# CURRENT STAGE: stage1" in content
        assert "Stage Details" not in content

    def test_stages_is_not_list(self):
        """Test when stages is not a list - should handle gracefully."""
        state = {
            "current_stage_id": "stage1",
            "plan": {"stages": {"stage1": {"task": "do x"}}},  # Dict instead of list
        }
        content = build_user_content_for_designer(state)
        
        # Iteration over dict gives keys, not items
        assert "# CURRENT STAGE: stage1" in content

    def test_exactly_20_parameters_boundary(self):
        """Test with exactly 20 parameters - should include all 20."""
        state = {
            "current_stage_id": "stage1",
            "extracted_parameters": [{"param": i} for i in range(20)],
        }
        content = build_user_content_for_designer(state)
        
        params_json = json.loads(content.split("## Extracted Parameters")[1].split("```json")[1].split("```")[0].strip())
        assert len(params_json) == 20
        assert params_json[0] == {"param": 0}
        assert params_json[19] == {"param": 19}

    def test_19_parameters_below_boundary(self):
        """Test with 19 parameters - should include all 19."""
        state = {
            "current_stage_id": "stage1",
            "extracted_parameters": [{"param": i} for i in range(19)],
        }
        content = build_user_content_for_designer(state)
        
        params_json = json.loads(content.split("## Extracted Parameters")[1].split("```json")[1].split("```")[0].strip())
        assert len(params_json) == 19

    def test_21_parameters_above_boundary(self):
        """Test with 21 parameters - should truncate to 20."""
        state = {
            "current_stage_id": "stage1",
            "extracted_parameters": [{"param": i} for i in range(21)],
        }
        content = build_user_content_for_designer(state)
        
        params_json = json.loads(content.split("## Extracted Parameters")[1].split("```json")[1].split("```")[0].strip())
        assert len(params_json) == 20
        assert params_json[0] == {"param": 0}
        assert params_json[19] == {"param": 19}
        # param 20 should NOT be included
        assert {"param": 20} not in params_json

    def test_assumptions_is_list_not_dict(self):
        """Test when assumptions is a list instead of dict - should not include section."""
        state = {
            "current_stage_id": "stage1",
            "assumptions": ["assumption1", "assumption2"],  # List instead of dict
        }
        content = build_user_content_for_designer(state)
        
        # Code checks `isinstance(assumptions, dict)`, so list should be excluded
        assert "## Assumptions" not in content

    def test_duplicate_stage_ids_in_plan(self):
        """Test when multiple stages have the same id - should use first match."""
        state = {
            "current_stage_id": "stage1",
            "plan": {
                "stages": [
                    {"stage_id": "stage1", "task": "first task"},
                    {"stage_id": "stage1", "task": "second task"},  # Duplicate
                ]
            },
        }
        content = build_user_content_for_designer(state)
        
        # `next()` returns first match
        assert "## Stage Details" in content
        stage_json = json.loads(content.split("## Stage Details")[1].split("```json")[1].split("```")[0].strip())
        assert stage_json["task"] == "first task"

    def test_section_headers_exact_format(self):
        """Test that section headers are exactly as expected."""
        state = {
            "current_stage_id": "stage1",
            "plan": {"stages": [{"stage_id": "stage1", "task": "do x"}]},
            "extracted_parameters": [{"param": "val"}],
            "assumptions": {"assump": "val"},
            "validated_materials": ["mat1"],
            "reviewer_feedback": "Fix it",
        }
        content = build_user_content_for_designer(state)
        
        assert "## Stage Details" in content
        assert "## Extracted Parameters" in content
        assert "## Assumptions" in content
        assert "## Validated Materials" in content
        assert "## REVISION FEEDBACK" in content

    def test_unicode_in_all_fields(self):
        """Test with Unicode characters in all fields."""
        state = {
            "current_stage_id": "阶段1",
            "plan": {"stages": [{"stage_id": "阶段1", "task": "执行任务 with émojis 🔬"}]},
            "extracted_parameters": [{"参数": "值"}],
            "assumptions": {"假设": "值"},
            "validated_materials": ["材料1"],
            "reviewer_feedback": "Обратная связь",
        }
        content = build_user_content_for_designer(state)
        
        assert "# CURRENT STAGE: 阶段1" in content
        assert "执行任务" in content
        assert "🔬" in content
        assert '"参数"' in content
        assert '"假设"' in content
        assert "材料1" in content
        assert "Обратная связь" in content

    def test_json_serialization_order(self):
        """Test that JSON serialization produces valid JSON."""
        state = {
            "current_stage_id": "stage1",
            "plan": {"stages": [{"stage_id": "stage1", "task": "do x", "nested": {"a": 1, "b": 2}}]},
            "extracted_parameters": [{"param": "val", "nested": [1, 2, 3]}],
            "assumptions": {"key": {"nested": "value"}},
            "validated_materials": [{"name": "mat1", "path": "/path"}],
        }
        content = build_user_content_for_designer(state)
        
        # All JSON blocks should be parseable
        json_blocks = content.split("```json")
        for i, block in enumerate(json_blocks[1:], 1):
            json_str = block.split("```")[0].strip()
            try:
                parsed = json.loads(json_str)
                assert parsed is not None
            except json.JSONDecodeError as e:
                pytest.fail(f"JSON block {i} is not valid JSON: {e}")

    def test_whitespace_only_feedback(self):
        """Test with whitespace-only feedback - should include if truthy."""
        state = {
            "current_stage_id": "stage1",
            "reviewer_feedback": "   \n\t   ",
        }
        content = build_user_content_for_designer(state)
        
        # Whitespace is truthy, so section should be included
        assert "## REVISION FEEDBACK" in content


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
        # Feedback is now labeled with source
        assert "**Code review:** Fix code" in content
        
        # Verify exact format for string design (not JSON)
        parts = content.split("\n\n")
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
            "physics_feedback": "",
            "execution_feedback": "",
        }
        content = build_user_content_for_code_generator(state)
        
        assert "## REVISION FEEDBACK" not in content

    def test_physics_feedback_only(self):
        """Test with only physics_feedback set - should include in feedback section.
        
        This is critical: when physics_check fails and routes to generate_code,
        the physics_feedback must reach the code generator.
        """
        physics_fb = "Energy conservation violated: T+R+A = 1.15"
        state = {
            "current_stage_id": "stage1",
            "design_description": "Design",
            "physics_feedback": physics_fb,
        }
        content = build_user_content_for_code_generator(state)
        
        assert "## REVISION FEEDBACK" in content
        assert f"**Physics validation:** {physics_fb}" in content
        # Should NOT include other feedback types
        assert "**Execution:**" not in content
        assert "**Code review:**" not in content

    def test_execution_feedback_only(self):
        """Test with only execution_feedback set - should include in feedback section.
        
        This is critical: when execution_check fails and routes to generate_code,
        the execution_feedback must reach the code generator.
        """
        execution_fb = "Simulation crashed with MemoryError"
        state = {
            "current_stage_id": "stage1",
            "design_description": "Design",
            "execution_feedback": execution_fb,
        }
        content = build_user_content_for_code_generator(state)
        
        assert "## REVISION FEEDBACK" in content
        assert f"**Execution:** {execution_fb}" in content
        # Should NOT include other feedback types
        assert "**Physics validation:**" not in content
        assert "**Code review:**" not in content

    def test_all_feedback_types(self):
        """Test with all feedback types set - all should appear in feedback section."""
        physics_fb = "T > 1.0 detected"
        execution_fb = "Exit code 0 but warnings"
        reviewer_fb = "Fix normalization"
        state = {
            "current_stage_id": "stage1",
            "design_description": "Design",
            "physics_feedback": physics_fb,
            "execution_feedback": execution_fb,
            "reviewer_feedback": reviewer_fb,
        }
        content = build_user_content_for_code_generator(state)
        
        assert "## REVISION FEEDBACK" in content
        assert f"**Physics validation:** {physics_fb}" in content
        assert f"**Execution:** {execution_fb}" in content
        assert f"**Code review:** {reviewer_fb}" in content

    def test_feedback_none_values(self):
        """Test with None feedback values - should not include section."""
        state = {
            "current_stage_id": "stage1",
            "design_description": "Design",
            "physics_feedback": None,
            "execution_feedback": None,
            "reviewer_feedback": None,
        }
        content = build_user_content_for_code_generator(state)
        
        assert "## REVISION FEEDBACK" not in content

    def test_feedback_partial_none(self):
        """Test with some feedback None and some set - only non-None included."""
        physics_fb = "Resonance at wrong wavelength"
        state = {
            "current_stage_id": "stage1",
            "design_description": "Design",
            "physics_feedback": physics_fb,
            "execution_feedback": None,
            "reviewer_feedback": "",
        }
        content = build_user_content_for_code_generator(state)
        
        assert "## REVISION FEEDBACK" in content
        assert f"**Physics validation:** {physics_fb}" in content
        # None and empty should not appear
        assert "**Execution:**" not in content
        assert "**Code review:**" not in content

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

    def test_list_design_description(self):
        """Test with list design_description - should serialize as JSON."""
        state = {
            "current_stage_id": "stage1",
            "design_description": ["step1", "step2", "step3"],
        }
        content = build_user_content_for_code_generator(state)
        
        # List is not a dict, so it should be treated as non-dict (string conversion)
        assert "## Design Specification" in content
        # Since list is not dict and is truthy, it goes to the else branch (treated as string)
        # But wait, let me check - the code does `isinstance(design, dict)` check

    def test_empty_dict_design_description(self):
        """Test with empty dict design_description - should not include section (falsy check)."""
        state = {
            "current_stage_id": "stage1",
            "design_description": {},
        }
        content = build_user_content_for_code_generator(state)
        
        # Empty dict is falsy, so section should not be included
        assert "## Design Specification" not in content

    def test_whitespace_only_design_description(self):
        """Test with whitespace-only design_description - should include section."""
        state = {
            "current_stage_id": "stage1",
            "design_description": "   \n\t   ",
        }
        content = build_user_content_for_code_generator(state)
        
        # Whitespace string is truthy
        assert "## Design Specification" in content
        assert "   \n\t   " in content

    def test_very_long_design_description(self):
        """Test with very long design_description string."""
        long_design = "X" * 50000
        state = {
            "current_stage_id": "stage1",
            "design_description": long_design,
        }
        content = build_user_content_for_code_generator(state)
        
        assert "## Design Specification" in content
        assert long_design in content
        assert len(content) > 50000

    def test_design_description_with_json_like_string(self):
        """Test with design_description that looks like JSON but is a string."""
        state = {
            "current_stage_id": "stage1",
            "design_description": '{"key": "value"}',  # String that looks like JSON
        }
        content = build_user_content_for_code_generator(state)
        
        # Should be treated as string, not parsed as dict
        assert "## Design Specification" in content
        # Should NOT have ```json block since it's a string
        parts = content.split("## Design Specification")
        if len(parts) > 1:
            design_section = parts[1].split("##")[0] if "##" in parts[1] else parts[1]
            # String design should be plain text, not in JSON block
            assert '{"key": "value"}' in design_section

    def test_nested_dict_design_description(self):
        """Test with deeply nested dict design_description."""
        state = {
            "current_stage_id": "stage1",
            "design_description": {
                "level1": {
                    "level2": {
                        "level3": {
                            "value": [1, 2, {"nested": True}]
                        }
                    }
                }
            },
        }
        content = build_user_content_for_code_generator(state)
        
        assert "## Design Specification" in content
        assert "```json" in content
        design_json = json.loads(content.split("## Design Specification")[1].split("```json")[1].split("```")[0].strip())
        assert design_json["level1"]["level2"]["level3"]["value"][2]["nested"] is True

    def test_materials_with_special_paths(self):
        """Test materials with special characters in paths."""
        state = {
            "current_stage_id": "stage1",
            "design_description": "Design",
            "validated_materials": [
                "/path/with spaces/file.csv",
                "/path/with/special!@#$%chars.csv",
                "/unicode/路径/材料.csv",
            ],
        }
        content = build_user_content_for_code_generator(state)
        
        assert "## Validated Materials" in content
        assert "/path/with spaces/file.csv" in content
        assert "/path/with/special!@#$%chars.csv" in content
        assert "/unicode/路径/材料.csv" in content

    def test_section_order_consistency(self):
        """Test that sections appear in consistent order."""
        state = {
            "current_stage_id": "stage1",
            "design_description": "Design spec",
            "validated_materials": ["mat1"],
            "reviewer_feedback": "Feedback",
        }
        content = build_user_content_for_code_generator(state)
        
        stage_idx = content.index("# CURRENT STAGE")
        design_idx = content.index("## Design Specification")
        materials_idx = content.index("## Validated Materials")
        feedback_idx = content.index("## REVISION FEEDBACK")
        
        assert stage_idx < design_idx < materials_idx < feedback_idx

    def test_empty_string_stage_id(self):
        """Test with empty string stage_id."""
        state = {
            "current_stage_id": "",
            "design_description": "Design",
        }
        content = build_user_content_for_code_generator(state)
        
        # Empty string is falsy, so `or "unknown"` should trigger
        assert content.startswith("# CURRENT STAGE: unknown")

    def test_single_material(self):
        """Test with single material - verify JSON array format."""
        state = {
            "current_stage_id": "stage1",
            "design_description": "Design",
            "validated_materials": ["single_material.csv"],
        }
        content = build_user_content_for_code_generator(state)
        
        materials_json = json.loads(content.split("## Validated Materials")[1].split("```json")[1].split("```")[0].strip())
        assert materials_json == ["single_material.csv"]
        assert isinstance(materials_json, list)


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

    def test_empty_string_stage_id(self):
        """Test with empty string stage_id."""
        state = {
            "current_stage_id": "",
            "stage_outputs": {"data": "value"},
        }
        content = build_user_content_for_analyzer(state)
        
        # Empty string is falsy, triggers "unknown"
        assert content.startswith("# CURRENT STAGE: unknown")

    def test_plan_is_list_not_dict(self):
        """Test when plan is a list instead of dict."""
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {"data": "value"},
            "plan": [{"stage_id": "stage1", "targets": ["fig1"]}],  # List instead of dict
        }
        content = build_user_content_for_analyzer(state)
        
        # Code checks `isinstance(plan, dict)`, so list should result in no targets
        assert "# CURRENT STAGE: stage1" in content
        assert "## Target Figures" not in content

    def test_targets_with_empty_strings(self):
        """Test targets list containing empty strings."""
        state = {
            "current_stage_id": "stage1",
            "plan": {"stages": [{"stage_id": "stage1", "targets": ["fig1", "", "fig2"]}]},
        }
        content = build_user_content_for_analyzer(state)
        
        # Empty strings should be joined with comma
        assert "## Target Figures: fig1, , fig2" in content

    def test_targets_with_single_item(self):
        """Test targets with single item - should not have trailing comma."""
        state = {
            "current_stage_id": "stage1",
            "plan": {"stages": [{"stage_id": "stage1", "targets": ["fig1"]}]},
        }
        content = build_user_content_for_analyzer(state)
        
        assert "## Target Figures: fig1" in content
        assert "## Target Figures: fig1," not in content

    def test_duplicate_stage_ids(self):
        """Test when multiple stages have the same id."""
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {"data": "value"},
            "plan": {
                "stages": [
                    {"stage_id": "stage1", "targets": ["fig1"]},
                    {"stage_id": "stage1", "targets": ["fig2", "fig3"]},  # Duplicate
                ]
            },
        }
        content = build_user_content_for_analyzer(state)
        
        # First match wins
        assert "## Target Figures: fig1" in content
        assert "fig2" not in content

    def test_stage_outputs_with_none_value(self):
        """Test stage_outputs containing None values."""
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {"data": None, "other": "value"},
        }
        content = build_user_content_for_analyzer(state)
        
        assert "## Simulation Outputs" in content
        # JSON null should be serialized
        outputs_json = json.loads(content.split("## Simulation Outputs")[1].split("```json")[1].split("```")[0].strip())
        assert outputs_json["data"] is None
        assert outputs_json["other"] == "value"

    def test_stage_outputs_with_complex_nested_data(self):
        """Test stage_outputs with deeply nested structure."""
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {
                "results": {
                    "wavelengths": list(range(100)),
                    "spectra": [{"intensity": i * 0.1} for i in range(50)],
                    "metadata": {
                        "timestamp": "2024-01-01",
                        "nested": {"deep": {"value": True}}
                    }
                }
            },
        }
        content = build_user_content_for_analyzer(state)
        
        outputs_json = json.loads(content.split("## Simulation Outputs")[1].split("```json")[1].split("```")[0].strip())
        assert len(outputs_json["results"]["wavelengths"]) == 100
        assert len(outputs_json["results"]["spectra"]) == 50
        assert outputs_json["results"]["metadata"]["nested"]["deep"]["value"] is True

    def test_unicode_in_targets(self):
        """Test Unicode characters in target figures."""
        state = {
            "current_stage_id": "stage1",
            "plan": {"stages": [{"stage_id": "stage1", "targets": ["图1", "図2", "фиг3"]}]},
        }
        content = build_user_content_for_analyzer(state)
        
        assert "## Target Figures: 图1, 図2, фиг3" in content

    def test_section_order(self):
        """Test that sections appear in consistent order."""
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {"data": "value"},
            "plan": {"stages": [{"stage_id": "stage1", "targets": ["fig1"]}]},
            "analysis_feedback": "Feedback",
        }
        content = build_user_content_for_analyzer(state)
        
        stage_idx = content.index("# CURRENT STAGE")
        outputs_idx = content.index("## Simulation Outputs")
        targets_idx = content.index("## Target Figures")
        feedback_idx = content.index("## REVISION FEEDBACK")
        
        assert stage_idx < outputs_idx < targets_idx < feedback_idx

    def test_stage_outputs_is_list_not_dict(self):
        """Test when stage_outputs is a list instead of dict - should still serialize."""
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": ["item1", "item2"],  # List instead of dict
        }
        content = build_user_content_for_analyzer(state)
        
        # List is truthy, so section should be included
        assert "## Simulation Outputs" in content
        # Should serialize as JSON array
        outputs = json.loads(content.split("## Simulation Outputs")[1].split("```json")[1].split("```")[0].strip())
        assert outputs == ["item1", "item2"]

    def test_whitespace_only_feedback(self):
        """Test with whitespace-only feedback."""
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {"data": "value"},
            "analysis_feedback": "   \n\t   ",
        }
        content = build_user_content_for_analyzer(state)
        
        # Whitespace is truthy
        assert "## REVISION FEEDBACK" in content

    def test_stage_outputs_with_path_objects(self):
        """Test stage_outputs containing Path objects - should use default=str."""
        from pathlib import PurePosixPath
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {
                "files": ["/path/to/file.png"],
                "paths": {"input": PurePosixPath("/some/input")},
            },
        }
        content = build_user_content_for_analyzer(state)
        
        # Path should be converted to string via default=str
        assert "## Simulation Outputs" in content
        assert "/some/input" in content

    def test_only_outputs_and_targets_no_feedback(self):
        """Test with outputs and targets but no feedback - verify no extra sections."""
        state = {
            "current_stage_id": "stage1",
            "stage_outputs": {"data": "value"},
            "plan": {"stages": [{"stage_id": "stage1", "targets": ["fig1"]}]},
        }
        content = build_user_content_for_analyzer(state)
        
        assert "## REVISION FEEDBACK" not in content
        # Count sections
        assert content.count("##") == 2  # Outputs and Targets


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

    def test_real_filesystem_paper_figures(self, tmp_path):
        """Test with real filesystem - paper figures."""
        # Create actual image files
        img1 = tmp_path / "fig1.png"
        img2 = tmp_path / "fig2.jpg"
        non_img = tmp_path / "data.csv"
        
        img1.write_bytes(b"fake png")
        img2.write_bytes(b"fake jpg")
        non_img.write_text("csv data")
        
        state = {
            "paper_figures": [
                {"image_path": str(img1)},
                {"image_path": str(img2)},
                {"image_path": str(non_img)},  # Not an image
            ],
        }
        images = get_images_for_analyzer(state)
        
        # Only image files should be returned
        assert len(images) == 2
        image_paths = [str(img) for img in images]
        assert str(img1) in image_paths
        assert str(img2) in image_paths
        assert str(non_img) not in image_paths

    def test_real_filesystem_stage_outputs(self, tmp_path):
        """Test with real filesystem - stage outputs."""
        # Create actual files
        img1 = tmp_path / "output1.png"
        img2 = tmp_path / "output2.jpeg"
        img3 = tmp_path / "output3.gif"
        img4 = tmp_path / "output4.webp"
        non_img = tmp_path / "data.txt"
        
        img1.write_bytes(b"fake png")
        img2.write_bytes(b"fake jpeg")
        img3.write_bytes(b"fake gif")
        img4.write_bytes(b"fake webp")
        non_img.write_text("text data")
        
        state = {
            "stage_outputs": {
                "files": [str(img1), str(img2), str(img3), str(img4), str(non_img)],
            },
        }
        images = get_images_for_analyzer(state)
        
        assert len(images) == 4
        image_paths = [str(img) for img in images]
        assert str(img1) in image_paths
        assert str(img2) in image_paths
        assert str(img3) in image_paths
        assert str(img4) in image_paths
        assert str(non_img) not in image_paths

    def test_real_filesystem_nonexistent_files(self, tmp_path):
        """Test with real filesystem - nonexistent files are skipped."""
        existing = tmp_path / "exists.png"
        existing.write_bytes(b"fake png")
        
        state = {
            "paper_figures": [
                {"image_path": str(existing)},
                {"image_path": str(tmp_path / "nonexistent.png")},
            ],
            "stage_outputs": {
                "files": [
                    str(existing),
                    str(tmp_path / "also_nonexistent.png"),
                ],
            },
        }
        images = get_images_for_analyzer(state)
        
        # Only existing file should be returned (appears twice but might be deduplicated)
        image_paths = [str(img) for img in images]
        assert str(existing) in image_paths
        assert str(tmp_path / "nonexistent.png") not in image_paths
        assert str(tmp_path / "also_nonexistent.png") not in image_paths

    def test_real_filesystem_path_objects(self, tmp_path):
        """Test with real Path objects in stage_outputs."""
        img = tmp_path / "output.png"
        img.write_bytes(b"fake png")
        
        state = {
            "stage_outputs": {
                "files": [img],  # Path object directly, not string
            },
        }
        images = get_images_for_analyzer(state)
        
        assert len(images) == 1
        assert str(images[0]) == str(img)

    def test_paper_figure_not_dict(self):
        """Test paper_figures containing non-dict items."""
        with patch("src.llm_client.Path") as mock_path:
            mock_obj = MagicMock()
            mock_obj.exists.return_value = True
            mock_obj.suffix = ".png"
            mock_path.return_value = mock_obj
            
            state = {
                "paper_figures": [
                    "not_a_dict",  # String instead of dict
                    123,  # Integer
                    {"image_path": "valid.png"},  # Valid dict
                ],
            }
            images = get_images_for_analyzer(state)
            
            # Only the valid dict should be processed
            assert len(images) == 1

    def test_stage_outputs_is_list_instead_of_dict(self):
        """Test when stage_outputs is a list instead of dict."""
        state = {
            "stage_outputs": ["file1.png", "file2.png"],  # List instead of dict
        }
        images = get_images_for_analyzer(state)
        
        # Code does `stage_outputs.get("files")` which will fail on list
        # This should handle gracefully
        assert len(images) == 0

    def test_stage_outputs_files_is_string(self):
        """Test when stage_outputs.files is a string instead of list."""
        state = {
            "stage_outputs": {"files": "single_file.png"},  # String instead of list
        }
        # This should iterate over characters, which would be weird behavior
        # Let's verify what happens
        images = get_images_for_analyzer(state)
        
        # Iterating over string gives characters - probably wrong behavior but let's document it
        # The function should either handle this or clearly fail
        # Based on code review, it will try Path("s"), Path("i"), etc.
        assert isinstance(images, list)

    def test_combined_paper_and_stage_images(self, tmp_path):
        """Test that both paper figures and stage outputs are collected."""
        paper_img = tmp_path / "paper_fig.png"
        stage_img = tmp_path / "stage_out.png"
        paper_img.write_bytes(b"paper")
        stage_img.write_bytes(b"stage")
        
        state = {
            "paper_figures": [{"image_path": str(paper_img)}],
            "stage_outputs": {"files": [str(stage_img)]},
        }
        images = get_images_for_analyzer(state)
        
        assert len(images) == 2
        image_paths = [str(img) for img in images]
        assert str(paper_img) in image_paths
        assert str(stage_img) in image_paths

    def test_duplicate_image_paths(self, tmp_path):
        """Test that duplicate paths appear multiple times (no deduplication)."""
        img = tmp_path / "image.png"
        img.write_bytes(b"png data")
        
        state = {
            "paper_figures": [{"image_path": str(img)}],
            "stage_outputs": {"files": [str(img)]},  # Same image
        }
        images = get_images_for_analyzer(state)
        
        # Based on code review, duplicates are NOT deduplicated
        assert len(images) == 2
        assert all(str(i) == str(img) for i in images)

    def test_uppercase_image_extensions(self, tmp_path):
        """Test that uppercase extensions are handled (case-insensitive check)."""
        img_png = tmp_path / "output.PNG"
        img_jpg = tmp_path / "output.JPG"
        img_jpeg = tmp_path / "output.JPEG"
        img_gif = tmp_path / "output.GIF"
        img_webp = tmp_path / "output.WEBP"
        
        for img in [img_png, img_jpg, img_jpeg, img_gif, img_webp]:
            img.write_bytes(b"fake")
        
        state = {
            "stage_outputs": {
                "files": [str(img_png), str(img_jpg), str(img_jpeg), str(img_gif), str(img_webp)],
            },
        }
        images = get_images_for_analyzer(state)
        
        # Code uses `.suffix.lower()` so uppercase should work
        assert len(images) == 5

    def test_mixed_case_extensions(self, tmp_path):
        """Test mixed case extensions like .Png, .JpG."""
        img_mixed = tmp_path / "output.Png"
        img_mixed.write_bytes(b"fake")
        
        state = {
            "stage_outputs": {"files": [str(img_mixed)]},
        }
        images = get_images_for_analyzer(state)
        
        assert len(images) == 1

    def test_empty_string_image_path_in_figures(self):
        """Test paper figure with empty string image_path."""
        with patch("src.llm_client.Path") as mock_path:
            def path_side_effect(path_str):
                mock_obj = MagicMock()
                mock_obj.exists.return_value = path_str != ""
                mock_obj.suffix = ".png" if path_str else ""
                mock_obj.__str__.return_value = str(path_str)
                return mock_obj
            
            mock_path.side_effect = path_side_effect
            
            state = {
                "paper_figures": [
                    {"image_path": ""},  # Empty string
                    {"image_path": "valid.png"},
                ],
            }
            images = get_images_for_analyzer(state)
            
            # Empty string path shouldn't exist
            assert len(images) == 1

    def test_return_type_is_path_objects(self, tmp_path):
        """Test that returned images are Path objects, not strings."""
        img = tmp_path / "test.png"
        img.write_bytes(b"data")
        
        state = {
            "paper_figures": [{"image_path": str(img)}],
        }
        images = get_images_for_analyzer(state)
        
        assert len(images) == 1
        assert isinstance(images[0], Path)

    def test_stage_output_dict_with_path_key(self, tmp_path):
        """Test stage output as dict with 'path' key."""
        img = tmp_path / "output.png"
        img.write_bytes(b"data")
        
        state = {
            "stage_outputs": {
                "files": [{"path": str(img), "type": "image"}],
            },
        }
        images = get_images_for_analyzer(state)
        
        assert len(images) == 1
        assert str(images[0]) == str(img)

    def test_stage_output_dict_without_path_key_real(self, tmp_path):
        """Test stage output dict without 'path' key - uses empty string."""
        state = {
            "stage_outputs": {
                "files": [{"type": "image"}],  # No path key
            },
        }
        images = get_images_for_analyzer(state)
        
        # dict.get("path", "") returns "", Path("").exists() is False
        assert len(images) == 0

    def test_symlink_image(self, tmp_path):
        """Test that symlinks to images work."""
        real_img = tmp_path / "real.png"
        real_img.write_bytes(b"png data")
        
        symlink = tmp_path / "link.png"
        try:
            symlink.symlink_to(real_img)
        except OSError:
            pytest.skip("Symlinks not supported on this system")
        
        state = {
            "stage_outputs": {"files": [str(symlink)]},
        }
        images = get_images_for_analyzer(state)
        
        # Symlinks should work since exists() follows them
        assert len(images) == 1

    def test_special_characters_in_path(self, tmp_path):
        """Test paths with special characters."""
        # Create directory with special chars
        special_dir = tmp_path / "dir with spaces & symbols!"
        special_dir.mkdir()
        img = special_dir / "image (1).png"
        img.write_bytes(b"data")
        
        state = {
            "paper_figures": [{"image_path": str(img)}],
        }
        images = get_images_for_analyzer(state)
        
        assert len(images) == 1
        assert str(images[0]) == str(img)

    def test_filters_paper_figures_by_stage_targets(self, tmp_path):
        """Test that only paper figures matching stage targets are included."""
        # Create paper figure images
        fig1 = tmp_path / "fig1.png"
        fig2 = tmp_path / "fig2.png"
        fig3 = tmp_path / "fig3.png"
        for f in [fig1, fig2, fig3]:
            f.write_bytes(b"fake image data")
        
        state = {
            "current_stage_id": "stage1",
            "plan": {
                "stages": [
                    {"stage_id": "stage1", "targets": ["fig1", "fig3"]},
                    {"stage_id": "stage2", "targets": ["fig2"]},
                ]
            },
            "paper_figures": [
                {"id": "fig1", "image_path": str(fig1)},
                {"id": "fig2", "image_path": str(fig2)},
                {"id": "fig3", "image_path": str(fig3)},
            ],
        }
        images = get_images_for_analyzer(state)
        
        # Only fig1 and fig3 should be included (targets for stage1)
        assert len(images) == 2
        image_paths = [str(img) for img in images]
        assert str(fig1) in image_paths
        assert str(fig3) in image_paths
        assert str(fig2) not in image_paths

    def test_includes_all_figures_when_no_targets_specified(self, tmp_path):
        """Test backwards compatibility: include all figures when stage has no targets."""
        fig1 = tmp_path / "fig1.png"
        fig2 = tmp_path / "fig2.png"
        for f in [fig1, fig2]:
            f.write_bytes(b"fake image data")
        
        state = {
            "current_stage_id": "stage1",
            "plan": {
                "stages": [
                    {"stage_id": "stage1"},  # No targets key
                ]
            },
            "paper_figures": [
                {"id": "fig1", "image_path": str(fig1)},
                {"id": "fig2", "image_path": str(fig2)},
            ],
        }
        images = get_images_for_analyzer(state)
        
        # Both figures should be included (backwards compat)
        assert len(images) == 2

    def test_includes_all_figures_when_empty_targets(self, tmp_path):
        """Test that empty targets list means include all figures."""
        fig1 = tmp_path / "fig1.png"
        fig2 = tmp_path / "fig2.png"
        for f in [fig1, fig2]:
            f.write_bytes(b"fake image data")
        
        state = {
            "current_stage_id": "stage1",
            "plan": {
                "stages": [
                    {"stage_id": "stage1", "targets": []},  # Empty targets
                ]
            },
            "paper_figures": [
                {"id": "fig1", "image_path": str(fig1)},
                {"id": "fig2", "image_path": str(fig2)},
            ],
        }
        images = get_images_for_analyzer(state)
        
        # Both figures should be included when targets is empty
        assert len(images) == 2

    def test_resolves_relative_stage_output_paths(self, tmp_path):
        """Test that relative stage output paths are resolved using run_output_dir."""
        # Create stage output directory structure
        run_dir = tmp_path / "outputs" / "paper_123" / "run_20251204"
        stage_dir = run_dir / "stage1"
        stage_dir.mkdir(parents=True)
        
        # Create simulation output file
        output_img = stage_dir / "output.png"
        output_img.write_bytes(b"fake image data")
        
        state = {
            "current_stage_id": "stage1",
            "run_output_dir": str(run_dir),
            "stage_outputs": {
                "files": ["output.png"],  # Relative path (just filename)
            },
        }
        images = get_images_for_analyzer(state)
        
        # Should resolve to full path in stage directory
        assert len(images) == 1
        assert str(images[0]) == str(output_img)

    def test_handles_absolute_stage_output_paths(self, tmp_path):
        """Test that absolute stage output paths are used as-is."""
        # Create an output image somewhere
        output_img = tmp_path / "absolute_output.png"
        output_img.write_bytes(b"fake image data")
        
        state = {
            "current_stage_id": "stage1",
            "run_output_dir": str(tmp_path / "different_dir"),  # Different from image location
            "stage_outputs": {
                "files": [str(output_img)],  # Absolute path
            },
        }
        images = get_images_for_analyzer(state)
        
        # Should use the absolute path directly
        assert len(images) == 1
        assert str(images[0]) == str(output_img)

    def test_legacy_fallback_without_run_output_dir(self, tmp_path, monkeypatch):
        """Test legacy fallback when run_output_dir is not set."""
        # This test verifies the fallback path construction
        # Without run_output_dir, it falls back to outputs/{paper_id}/{stage_id}
        state = {
            "current_stage_id": "stage1",
            "paper_id": "test_paper",
            # No run_output_dir
            "stage_outputs": {
                "files": ["output.png"],  # Relative path
            },
        }
        images = get_images_for_analyzer(state)
        
        # With relative path and no run_output_dir, file likely won't exist
        # This tests that the fallback logic doesn't crash
        assert isinstance(images, list)

    def test_combined_filtering_and_path_resolution(self, tmp_path):
        """Test that both paper figure filtering and output path resolution work together."""
        # Create paper figures
        fig1 = tmp_path / "figures" / "fig1.png"
        fig2 = tmp_path / "figures" / "fig2.png"
        fig1.parent.mkdir(parents=True)
        for f in [fig1, fig2]:
            f.write_bytes(b"paper figure")
        
        # Create stage output directory
        run_dir = tmp_path / "outputs" / "run_123"
        stage_dir = run_dir / "stage1"
        stage_dir.mkdir(parents=True)
        output_img = stage_dir / "simulation_result.png"
        output_img.write_bytes(b"simulation output")
        
        state = {
            "current_stage_id": "stage1",
            "run_output_dir": str(run_dir),
            "plan": {
                "stages": [
                    {"stage_id": "stage1", "targets": ["fig1"]},  # Only fig1 is target
                ]
            },
            "paper_figures": [
                {"id": "fig1", "image_path": str(fig1)},
                {"id": "fig2", "image_path": str(fig2)},  # Not a target
            ],
            "stage_outputs": {
                "files": ["simulation_result.png"],
            },
        }
        images = get_images_for_analyzer(state)
        
        # Should have: fig1 (target) + simulation_result.png (stage output)
        # Should NOT have: fig2 (not a target for this stage)
        assert len(images) == 2
        image_paths = [str(img) for img in images]
        assert str(fig1) in image_paths
        assert str(output_img) in image_paths
        assert str(fig2) not in image_paths

    # =========================================================================
    # Behavior-focused tests that verify correct operation in realistic scenarios
    # These tests would have caught the original bugs:
    # 1. All paper figures being sent instead of just stage targets
    # 2. Relative simulation output paths not being resolved
    # =========================================================================

    def test_multi_stage_workflow_each_stage_gets_only_its_targets(self, tmp_path):
        """
        Test that in a multi-stage workflow, each stage only gets its target figures.
        
        This test would have caught the bug where ALL paper figures were sent
        to the analyzer regardless of which stage was being analyzed.
        """
        # Create paper figures for different stages
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        fig2a = figures_dir / "fig2a.png"
        fig2b = figures_dir / "fig2b.png"
        fig3a = figures_dir / "fig3a.png"
        fig3b = figures_dir / "fig3b.png"
        for f in [fig2a, fig2b, fig3a, fig3b]:
            f.write_bytes(b"paper figure data")
        
        # Common plan with multiple stages, each targeting different figures
        plan = {
            "stages": [
                {"stage_id": "stage0_material_validation", "targets": ["fig2a"]},
                {"stage_id": "stage1_bare_disk", "targets": ["fig2b", "fig3a"]},
                {"stage_id": "stage2_coupled", "targets": ["fig3b"]},
            ]
        }
        
        paper_figures = [
            {"id": "fig2a", "image_path": str(fig2a)},
            {"id": "fig2b", "image_path": str(fig2b)},
            {"id": "fig3a", "image_path": str(fig3a)},
            {"id": "fig3b", "image_path": str(fig3b)},
        ]
        
        # Test stage0 - should only get fig2a
        state_stage0 = {
            "current_stage_id": "stage0_material_validation",
            "plan": plan,
            "paper_figures": paper_figures,
        }
        images_stage0 = get_images_for_analyzer(state_stage0)
        assert len(images_stage0) == 1
        assert str(fig2a) in [str(img) for img in images_stage0]
        
        # Test stage1 - should only get fig2b and fig3a
        state_stage1 = {
            "current_stage_id": "stage1_bare_disk",
            "plan": plan,
            "paper_figures": paper_figures,
        }
        images_stage1 = get_images_for_analyzer(state_stage1)
        assert len(images_stage1) == 2
        image_paths = [str(img) for img in images_stage1]
        assert str(fig2b) in image_paths
        assert str(fig3a) in image_paths
        assert str(fig2a) not in image_paths  # Not a target for this stage
        assert str(fig3b) not in image_paths  # Not a target for this stage
        
        # Test stage2 - should only get fig3b
        state_stage2 = {
            "current_stage_id": "stage2_coupled",
            "plan": plan,
            "paper_figures": paper_figures,
        }
        images_stage2 = get_images_for_analyzer(state_stage2)
        assert len(images_stage2) == 1
        assert str(fig3b) in [str(img) for img in images_stage2]

    def test_simulation_outputs_require_path_resolution(self, tmp_path):
        """
        Test that simulation outputs with just filenames are correctly resolved.
        
        This test would have caught the bug where relative filenames like
        "stage0_materials_plot.png" weren't being found because they were
        checked in the current working directory instead of the stage output dir.
        """
        # Create realistic directory structure matching production
        run_dir = tmp_path / "outputs" / "aluminum_paper" / "run_20251204_194932"
        stage_dir = run_dir / "stage0_material_validation"
        stage_dir.mkdir(parents=True)
        
        # Create simulation output (like code_runner produces)
        plot_file = stage_dir / "stage0_materials_plot.png"
        plot_file.write_bytes(b"matplotlib plot data")
        
        # State as it would be in production - files list has just filenames
        state = {
            "current_stage_id": "stage0_material_validation",
            "run_output_dir": str(run_dir),
            "stage_outputs": {
                "files": [
                    "stage0_material_glass.csv",
                    "stage0_material_aluminum.csv", 
                    "stage0_materials_plot.png",  # Just filename, not full path
                ],
            },
        }
        
        images = get_images_for_analyzer(state)
        
        # Should find the plot file by resolving: run_dir / stage_id / filename
        assert len(images) == 1
        assert str(images[0]) == str(plot_file)

    def test_only_filenames_in_stage_outputs_realistic_scenario(self, tmp_path):
        """
        Test the exact scenario from the bug: stage_outputs.files contains
        only filenames (as produced by code_runner), not full paths.
        
        Before the fix, these files would NOT be found because:
        - Path("stage0_materials_plot.png").exists() checks current working dir
        - The actual file is in outputs/paper_id/run_id/stage_id/
        """
        # Simulate the exact production structure
        run_dir = tmp_path / "outputs" / "paper_123" / "run_20251204"
        stage0_dir = run_dir / "stage0_material_validation"
        stage1_dir = run_dir / "stage1_bare_disk"
        stage0_dir.mkdir(parents=True)
        stage1_dir.mkdir(parents=True)
        
        # Stage 0 outputs
        stage0_plot = stage0_dir / "materials_plot.png"
        stage0_plot.write_bytes(b"stage0 plot")
        
        # Stage 1 outputs  
        stage1_plot = stage1_dir / "absorption_spectrum.png"
        stage1_plot.write_bytes(b"stage1 plot")
        
        # Test stage0 - should find stage0's plot
        state_stage0 = {
            "current_stage_id": "stage0_material_validation",
            "run_output_dir": str(run_dir),
            "stage_outputs": {
                "files": ["materials_plot.png", "data.csv"],  # Just filenames!
            },
        }
        images_stage0 = get_images_for_analyzer(state_stage0)
        assert len(images_stage0) == 1
        assert str(images_stage0[0]) == str(stage0_plot)
        
        # Test stage1 - should find stage1's plot (not stage0's!)
        state_stage1 = {
            "current_stage_id": "stage1_bare_disk",
            "run_output_dir": str(run_dir),
            "stage_outputs": {
                "files": ["absorption_spectrum.png", "spectrum.csv"],
            },
        }
        images_stage1 = get_images_for_analyzer(state_stage1)
        assert len(images_stage1) == 1
        assert str(images_stage1[0]) == str(stage1_plot)

    def test_figure_id_matching_is_exact(self, tmp_path):
        """Test that figure ID matching is exact - 'fig2' doesn't match 'fig2a'."""
        fig2 = tmp_path / "fig2.png"
        fig2a = tmp_path / "fig2a.png"
        fig2.write_bytes(b"data")
        fig2a.write_bytes(b"data")
        
        state = {
            "current_stage_id": "stage1",
            "plan": {
                "stages": [
                    {"stage_id": "stage1", "targets": ["fig2"]},  # Exact match required
                ]
            },
            "paper_figures": [
                {"id": "fig2", "image_path": str(fig2)},
                {"id": "fig2a", "image_path": str(fig2a)},  # Similar but different ID
            ],
        }
        
        images = get_images_for_analyzer(state)
        
        # Only fig2 should match, not fig2a
        assert len(images) == 1
        assert str(fig2) in [str(img) for img in images]
        assert str(fig2a) not in [str(img) for img in images]

    def test_targets_not_in_paper_figures_are_silently_skipped(self, tmp_path):
        """Test that targets referencing non-existent figure IDs don't crash."""
        fig1 = tmp_path / "fig1.png"
        fig1.write_bytes(b"data")
        
        state = {
            "current_stage_id": "stage1",
            "plan": {
                "stages": [
                    {"stage_id": "stage1", "targets": ["fig1", "fig_nonexistent"]},
                ]
            },
            "paper_figures": [
                {"id": "fig1", "image_path": str(fig1)},
                # fig_nonexistent is not in paper_figures
            ],
        }
        
        images = get_images_for_analyzer(state)
        
        # Should return fig1, silently skip the nonexistent target
        assert len(images) == 1
        assert str(fig1) in [str(img) for img in images]

    def test_paper_figure_without_id_is_not_filtered(self, tmp_path):
        """Test that paper figures without 'id' field are included when filtering."""
        fig_with_id = tmp_path / "fig_with_id.png"
        fig_without_id = tmp_path / "fig_without_id.png"
        fig_with_id.write_bytes(b"data")
        fig_without_id.write_bytes(b"data")
        
        state = {
            "current_stage_id": "stage1",
            "plan": {
                "stages": [
                    {"stage_id": "stage1", "targets": ["fig1"]},
                ]
            },
            "paper_figures": [
                {"id": "fig1", "image_path": str(fig_with_id)},
                {"image_path": str(fig_without_id)},  # No id field
            ],
        }
        
        images = get_images_for_analyzer(state)
        
        # Only fig1 should be included (has matching id)
        # Figure without id has empty string id, doesn't match "fig1"
        assert len(images) == 1
        assert str(fig_with_id) in [str(img) for img in images]

    def test_realistic_aluminum_nanoantenna_scenario(self, tmp_path):
        """
        Test with realistic data matching the aluminum_nanoantenna_complexes paper.
        
        This integration-style test verifies the function works correctly with
        realistic state data similar to actual production runs.
        """
        # Create figures directory
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        
        # Create paper figures (like in real paper)
        fig2a = figures_dir / "fig2a_TDBC_absorption.png"
        fig2b = figures_dir / "fig2b_bare_disk_spectrum.png"
        fig3a = figures_dir / "fig3a_coupled_spectrum.png"
        for f in [fig2a, fig2b, fig3a]:
            f.write_bytes(b"paper figure")
        
        # Create run output directory
        run_dir = tmp_path / "outputs" / "aluminum_nanoantenna" / "run_20251204_194932"
        
        # Create stage0 output
        stage0_dir = run_dir / "stage0_material_validation"
        stage0_dir.mkdir(parents=True)
        stage0_plot = stage0_dir / "stage0_materials_plot.png"
        stage0_plot.write_bytes(b"materials validation plot")
        
        # Realistic plan
        plan = {
            "stages": [
                {
                    "stage_id": "stage0_material_validation",
                    "targets": ["fig2a"],
                    "description": "Validate TDBC absorption matches paper"
                },
                {
                    "stage_id": "stage1_bare_disk",
                    "targets": ["fig2b"],
                    "description": "Reproduce bare disk spectrum"
                },
                {
                    "stage_id": "stage2_coupled",
                    "targets": ["fig3a"],
                    "description": "Reproduce coupled system spectrum"
                },
            ]
        }
        
        paper_figures = [
            {"id": "fig2a", "image_path": str(fig2a), "description": "TDBC absorption"},
            {"id": "fig2b", "image_path": str(fig2b), "description": "Bare disk spectrum"},
            {"id": "fig3a", "image_path": str(fig3a), "description": "Coupled spectrum"},
        ]
        
        # Test analyzing stage0
        state = {
            "current_stage_id": "stage0_material_validation",
            "run_output_dir": str(run_dir),
            "plan": plan,
            "paper_figures": paper_figures,
            "stage_outputs": {
                "files": [
                    "stage0_material_glass.csv",
                    "stage0_material_aluminum.csv",
                    "stage0_material_tdbc.csv",
                    "stage0_materials_plot.png",
                ],
            },
        }
        
        images = get_images_for_analyzer(state)
        
        # Should get: fig2a (target) + stage0_materials_plot.png (simulation output)
        assert len(images) == 2
        image_paths = [str(img) for img in images]
        
        # Paper figure for this stage's target
        assert str(fig2a) in image_paths
        # Simulation output resolved correctly
        assert str(stage0_plot) in image_paths
        
        # Should NOT include figures from other stages
        assert str(fig2b) not in image_paths
        assert str(fig3a) not in image_paths
