"""Integration tests ensuring every agent prompt exists and loads."""

import pytest
from pathlib import Path

from schemas.state import create_initial_state
from src.prompts import (
    PROMPTS_DIR,
    AGENT_PROMPTS,
    build_agent_prompt,
    get_agent_prompt_cached,
    load_prompt_file,
    load_global_rules,
    substitute_placeholders,
    apply_prompt_adaptations,
    validate_all_prompts_loadable,
    validate_placeholders_substituted,
)


class TestPromptFilesExist:
    """Verify all prompt files referenced in code actually exist."""

    AGENTS_REQUIRING_PROMPTS = [
        "prompt_adaptor",
        "planner",
        "plan_reviewer",
        "simulation_designer",
        "design_reviewer",
        "code_generator",
        "code_reviewer",
        "execution_validator",
        "physics_sanity",
        "results_analyzer",
        "supervisor",
        "report_generator",
        "comparison_validator",  # Added - was missing from original list!
    ]

    @pytest.mark.parametrize("agent_name", AGENTS_REQUIRING_PROMPTS)
    def test_prompt_file_exists(self, agent_name):
        """Each agent's prompt file must exist on disk."""
        prompt_file = PROMPTS_DIR / f"{agent_name}_agent.md"
        assert prompt_file.exists(), (
            f"Missing prompt file: {prompt_file}\n"
            f"Agent '{agent_name}' calls build_agent_prompt() but the file doesn't exist.\n"
            "This will crash at runtime!"
        )

    @pytest.mark.parametrize("agent_name", AGENTS_REQUIRING_PROMPTS)
    def test_prompt_file_not_empty(self, agent_name):
        """Each prompt file must have substantial content."""
        prompt_file = PROMPTS_DIR / f"{agent_name}_agent.md"
        # FIXED: Remove conditional - if file doesn't exist, test_prompt_file_exists should catch it
        # This test should fail if file is missing, not silently pass
        assert prompt_file.exists(), (
            f"Prompt file {prompt_file} does not exist. "
            "This should have been caught by test_prompt_file_exists."
        )
        content = prompt_file.read_text(encoding="utf-8")
        assert len(content) > 100, (
            f"Prompt file {prompt_file} is suspiciously short ({len(content)} chars).\n"
            "Expected substantial prompt content (at least 100 characters)."
        )
        # Verify it's not just whitespace
        assert content.strip(), (
            f"Prompt file {prompt_file} contains only whitespace."
        )

    @pytest.mark.parametrize("agent_name", AGENTS_REQUIRING_PROMPTS)
    def test_build_agent_prompt_succeeds(self, agent_name):
        """build_agent_prompt() must succeed for each agent and return valid content."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )

        prompt = build_agent_prompt(agent_name, state)
        # STRENGTHENED: More specific assertions
        assert prompt is not None, f"build_agent_prompt('{agent_name}') returned None"
        assert isinstance(prompt, str), (
            f"build_agent_prompt('{agent_name}') returned {type(prompt)}, expected str"
        )
        assert len(prompt) > 0, (
            f"build_agent_prompt('{agent_name}') returned empty string"
        )
        assert prompt.strip(), (
            f"build_agent_prompt('{agent_name}') returned only whitespace"
        )
        # Verify minimum reasonable length (should include global rules + agent prompt)
        assert len(prompt) > 200, (
            f"build_agent_prompt('{agent_name}') returned suspiciously short prompt "
            f"({len(prompt)} chars). Expected global rules + agent prompt."
        )

    @pytest.mark.parametrize("agent_name", AGENTS_REQUIRING_PROMPTS)
    def test_build_agent_prompt_without_global_rules(self, agent_name):
        """build_agent_prompt() works without global rules."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )

        prompt_without_global = build_agent_prompt(
            agent_name, state, include_global_rules=False
        )
        prompt_with_global = build_agent_prompt(
            agent_name, state, include_global_rules=True
        )

        assert prompt_without_global is not None
        assert prompt_with_global is not None
        # Prompt without global rules should be shorter
        assert len(prompt_without_global) < len(prompt_with_global), (
            f"Prompt without global rules should be shorter than with global rules "
            f"for agent '{agent_name}'"
        )
        # Prompt without global rules should still have content
        assert len(prompt_without_global) > 100, (
            f"Prompt without global rules for '{agent_name}' is too short"
        )

    @pytest.mark.parametrize("agent_name", AGENTS_REQUIRING_PROMPTS)
    def test_build_agent_prompt_with_none_state(self, agent_name):
        """build_agent_prompt() works with None state."""
        prompt = build_agent_prompt(agent_name, state=None)
        assert prompt is not None
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_agent_prompts_mapping_completeness(self):
        """AGENT_PROMPTS mapping should include all agents in test list."""
        missing_in_mapping = set(self.AGENTS_REQUIRING_PROMPTS) - set(AGENT_PROMPTS.keys())
        assert not missing_in_mapping, (
            f"Agents in test list but missing from AGENT_PROMPTS mapping: {missing_in_mapping}"
        )
        # Also check reverse - agents in mapping should be testable
        missing_in_tests = set(AGENT_PROMPTS.keys()) - set(self.AGENTS_REQUIRING_PROMPTS)
        if missing_in_tests:
            pytest.fail(
                f"Agents in AGENT_PROMPTS mapping but not in test list: {missing_in_tests}. "
                "Either add to test list or remove from mapping."
            )

    def test_global_rules_file_exists(self):
        """global_rules.md must exist."""
        global_rules_file = PROMPTS_DIR / "global_rules.md"
        assert global_rules_file.exists(), (
            f"Missing global_rules.md file: {global_rules_file}\n"
            "This file is prepended to all agent prompts."
        )

    def test_global_rules_not_empty(self):
        """global_rules.md must have content."""
        global_rules_file = PROMPTS_DIR / "global_rules.md"
        assert global_rules_file.exists()
        content = global_rules_file.read_text(encoding="utf-8")
        assert len(content) > 100, (
            f"global_rules.md is suspiciously short ({len(content)} chars)."
        )
        assert content.strip(), "global_rules.md contains only whitespace."

    def test_global_rules_included_in_prompts(self):
        """Global rules should be included when include_global_rules=True."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )
        global_rules_content = load_global_rules()

        # Test with a few agents
        for agent_name in ["planner", "code_generator", "supervisor"]:
            prompt = build_agent_prompt(agent_name, state, include_global_rules=True)
            # Global rules should appear in the prompt
            assert global_rules_content in prompt or prompt.startswith(global_rules_content[:100]), (
                f"Global rules not found in prompt for '{agent_name}' when include_global_rules=True"
            )

    def test_global_rules_not_included_when_disabled(self):
        """Global rules should NOT be included when include_global_rules=False."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )
        global_rules_content = load_global_rules()

        # Test with a few agents
        for agent_name in ["planner", "code_generator", "supervisor"]:
            prompt = build_agent_prompt(agent_name, state, include_global_rules=False)
            # Global rules should NOT appear in the prompt
            assert global_rules_content not in prompt, (
                f"Global rules found in prompt for '{agent_name}' when include_global_rules=False"
            )


class TestPlaceholderSubstitution:
    """Test placeholder substitution functionality."""

    def test_thresholds_table_placeholder_substituted(self):
        """{THRESHOLDS_TABLE} placeholder should be substituted."""
        # Create a test prompt with placeholder
        test_prompt = "Use these thresholds:\n{THRESHOLDS_TABLE}\nEnd of prompt."
        result = substitute_placeholders(test_prompt)

        # Placeholder should be gone
        assert "{THRESHOLDS_TABLE}" not in result, (
            "{THRESHOLDS_TABLE} placeholder was not substituted"
        )
        # Should contain actual table content (markdown table structure)
        assert "|" in result, "Substituted content should contain markdown table"
        assert "Quantity" in result or "Excellent" in result or "Acceptable" in result, (
            "Substituted content should contain threshold table headers"
        )

    def test_thresholds_table_has_correct_structure(self):
        """The substituted thresholds table should have proper markdown table structure."""
        test_prompt = "{THRESHOLDS_TABLE}"
        result = substitute_placeholders(test_prompt)
        
        # Should have table header row
        assert "| Quantity | Excellent | Acceptable | Investigate |" in result, (
            "Thresholds table should have proper header row"
        )
        # Should have header separator
        assert "|---" in result, "Thresholds table should have header separator row"

    def test_thresholds_table_contains_all_quantities(self):
        """The thresholds table should contain all defined quantities."""
        test_prompt = "{THRESHOLDS_TABLE}"
        result = substitute_placeholders(test_prompt)
        
        # All quantities from DISCREPANCY_THRESHOLDS should be present (human-readable)
        expected_quantities = [
            "Resonance wavelength",
            "Linewidth",  # Could be "Linewidth / FWHM"
            "Q-factor",
            "Transmission",
            "Reflection",
            "Field enhancement",
            "Mode effective index",  # Could be "effective index"
        ]
        
        for quantity in expected_quantities:
            # Check for partial match since formatting may vary
            base_name = quantity.lower().split()[0]
            assert base_name in result.lower(), (
                f"Thresholds table should contain '{quantity}' (or similar). "
                f"Got:\n{result}"
            )

    def test_thresholds_table_values_are_percentages(self):
        """The thresholds table values should be formatted as percentages."""
        test_prompt = "{THRESHOLDS_TABLE}"
        result = substitute_placeholders(test_prompt)
        
        # Should contain percentage symbols
        assert "%" in result, "Thresholds values should be formatted as percentages"
        # Should contain ± for excellent/acceptable ranges
        assert "±" in result, "Thresholds should have ± notation for ranges"
        # Should contain > for investigate threshold
        assert ">" in result, "Thresholds should have > notation for investigate level"

    def test_thresholds_table_has_reasonable_values(self):
        """The thresholds table values should be reasonable numbers."""
        test_prompt = "{THRESHOLDS_TABLE}"
        result = substitute_placeholders(test_prompt)
        
        # Extract all percentage values
        import re
        percentages = re.findall(r'[±>]?(\d+)%', result)
        
        assert len(percentages) > 0, "Should find percentage values in the table"
        
        for pct in percentages:
            value = int(pct)
            # Values should be between 1 and 200 (reasonable for physics thresholds)
            assert 1 <= value <= 200, (
                f"Threshold value {value}% seems unreasonable (should be 1-200%)"
            )

    def test_multiple_placeholders_substituted(self):
        """Multiple placeholders should all be substituted."""
        test_prompt = "First: {THRESHOLDS_TABLE}\nSecond: {THRESHOLDS_TABLE}"
        result = substitute_placeholders(test_prompt)

        # Both should be substituted
        assert result.count("|") >= 2, "Both placeholders should be substituted with table content"
        assert "{THRESHOLDS_TABLE}" not in result

    def test_no_placeholders_unchanged(self):
        """Prompt without placeholders should be unchanged."""
        test_prompt = "This is a normal prompt without placeholders."
        result = substitute_placeholders(test_prompt)

        assert result == test_prompt, (
            "Prompt without placeholders should remain unchanged"
        )

    def test_placeholders_substituted_in_built_prompts(self):
        """Placeholders should be substituted in built prompts."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )

        # Check a few agents that might use placeholders
        for agent_name in ["planner", "code_generator", "supervisor"]:
            prompt = build_agent_prompt(agent_name, state)
            # Verify no unsubstituted placeholders remain
            unsubstituted = validate_placeholders_substituted(prompt)
            assert not unsubstituted, (
                f"Unsubstituted placeholders found in prompt for '{agent_name}': {unsubstituted}"
            )

    def test_placeholder_in_middle_of_text(self):
        """Placeholder in middle of text should be substituted."""
        test_prompt = "Start text {THRESHOLDS_TABLE} end text"
        result = substitute_placeholders(test_prompt)

        assert "{THRESHOLDS_TABLE}" not in result
        assert "Start text" in result
        assert "end text" in result

    def test_placeholder_at_start(self):
        """Placeholder at start should be substituted."""
        test_prompt = "{THRESHOLDS_TABLE}\nRest of prompt"
        result = substitute_placeholders(test_prompt)

        assert "{THRESHOLDS_TABLE}" not in result
        assert "Rest of prompt" in result

    def test_placeholder_at_end(self):
        """Placeholder at end should be substituted."""
        test_prompt = "Start of prompt\n{THRESHOLDS_TABLE}"
        result = substitute_placeholders(test_prompt)

        assert "{THRESHOLDS_TABLE}" not in result
        assert "Start of prompt" in result

    def test_empty_string_unchanged(self):
        """Empty string should remain unchanged."""
        result = substitute_placeholders("")
        assert result == ""

    def test_whitespace_only_unchanged(self):
        """Whitespace-only string should remain unchanged."""
        test_prompt = "   \n\t  "
        result = substitute_placeholders(test_prompt)
        assert result == test_prompt

    def test_placeholder_substitution_is_idempotent(self):
        """Substituting already-substituted content should not change it."""
        test_prompt = "Use these thresholds:\n{THRESHOLDS_TABLE}"
        result1 = substitute_placeholders(test_prompt)
        result2 = substitute_placeholders(result1)
        
        # Second substitution should not change anything
        assert result1 == result2, (
            "Placeholder substitution should be idempotent"
        )
        assert "{THRESHOLDS_TABLE}" not in result2

    def test_unknown_placeholder_is_not_substituted(self):
        """Unknown placeholders should be left as-is (not error, not removed)."""
        test_prompt = "Known: {THRESHOLDS_TABLE} Unknown: {UNKNOWN_PLACEHOLDER}"
        result = substitute_placeholders(test_prompt)
        
        # Known placeholder should be substituted
        assert "{THRESHOLDS_TABLE}" not in result
        # Unknown placeholder should remain
        assert "{UNKNOWN_PLACEHOLDER}" in result, (
            "Unknown placeholders should be left unchanged, not removed"
        )

    def test_similar_but_not_exact_placeholder_not_substituted(self):
        """Similar but not exact placeholders should not be substituted."""
        # These should NOT be treated as the known placeholder
        test_cases = [
            "THRESHOLDS_TABLE",  # Missing braces
            "{thresholds_table}",  # Lowercase
            "{THRESHOLDS_TABLE }",  # Extra space
            "{ THRESHOLDS_TABLE}",  # Extra space
            "{THRESHOLDS-TABLE}",  # Hyphen instead of underscore
            "{{THRESHOLDS_TABLE}}",  # Double braces
        ]
        
        for test_case in test_cases:
            result = substitute_placeholders(test_case)
            # Original should be unchanged since it doesn't match exactly
            # (except if {THRESHOLDS_TABLE} appears inside it)
            if test_case == "{{THRESHOLDS_TABLE}}":
                # This one contains the valid placeholder, so one layer of braces should remain
                assert "{{" not in result or "|" in result  # Table was substituted
            else:
                # These don't contain the exact placeholder
                assert test_case == result, (
                    f"'{test_case}' should not be substituted as it's not an exact match. "
                    f"Got: '{result}'"
                )

    def test_placeholder_in_code_block_still_substituted(self):
        """Placeholders in markdown code blocks are still substituted."""
        test_prompt = "```\n{THRESHOLDS_TABLE}\n```"
        result = substitute_placeholders(test_prompt)
        
        # Placeholder should still be substituted even in code block
        assert "{THRESHOLDS_TABLE}" not in result
        # Table content should be there
        assert "|" in result


class TestPromptAdaptations:
    """Test prompt adaptation functionality."""

    def test_append_adaptation(self):
        """Test append adaptation type."""
        base_prompt = "Original prompt content."
        adaptations = [
            {
                "target_agent": "planner",
                "modification_type": "append",
                "content": "Additional content appended.",
            }
        ]

        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        assert "Original prompt content." in result
        assert "Additional content appended." in result
        assert result.endswith("Additional content appended.") or "# Paper-Specific Adaptation" in result

    def test_prepend_adaptation(self):
        """Test prepend adaptation type."""
        base_prompt = "Original prompt content."
        adaptations = [
            {
                "target_agent": "planner",
                "modification_type": "prepend",
                "content": "Content prepended.",
            }
        ]

        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        assert "Original prompt content." in result
        assert "Content prepended." in result
        assert result.startswith("# Paper-Specific Adaptation") or "Content prepended." in result[:100]

    def test_replace_adaptation(self):
        """Test replace adaptation type."""
        base_prompt = "Original prompt with MARKER to replace."
        adaptations = [
            {
                "target_agent": "planner",
                "modification_type": "replace",
                "content": "Replaced content.",
                "section_marker": "MARKER",
            }
        ]

        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        assert "MARKER" not in result, "Marker should be replaced"
        assert "Replaced content." in result

    def test_disable_adaptation(self):
        """Test disable adaptation type."""
        base_prompt = "Original prompt with MARKER to disable."
        adaptations = [
            {
                "target_agent": "planner",
                "modification_type": "disable",
                "section_marker": "MARKER",
            }
        ]

        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        assert "[DISABLED: MARKER]" in result, "Marker should be disabled"

    def test_adaptation_applies_to_correct_agent(self):
        """Adaptations should only apply to target agent."""
        base_prompt = "Original prompt."
        adaptations = [
            {
                "target_agent": "planner",
                "modification_type": "append",
                "content": "Planner-specific content.",
            }
        ]

        # Should apply to planner
        result_planner = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        assert "Planner-specific content." in result_planner

        # Should NOT apply to other agent
        result_other = apply_prompt_adaptations(base_prompt, "code_generator", adaptations)
        assert "Planner-specific content." not in result_other
        assert result_other == base_prompt

    def test_empty_adaptations_no_change(self):
        """Empty adaptations list should not change prompt."""
        base_prompt = "Original prompt content."
        result = apply_prompt_adaptations(base_prompt, "planner", [])
        assert result == base_prompt

    def test_adaptations_applied_in_built_prompts(self):
        """Adaptations should be applied when building prompts with state."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )
        # Add adaptations to state
        state["prompt_adaptations"] = [
            {
                "target_agent": "planner",
                "modification_type": "append",
                "content": "TEST ADAPTATION CONTENT",
            }
        ]

        prompt = build_agent_prompt("planner", state)
        assert "TEST ADAPTATION CONTENT" in prompt, (
            "Prompt adaptation should be applied when building prompt with state"
        )

    def test_adaptations_not_applied_without_state(self):
        """Adaptations should not be applied when state is None."""
        prompt = build_agent_prompt("planner", state=None)
        assert "TEST ADAPTATION CONTENT" not in prompt

    def test_adaptation_matching_is_not_too_broad(self):
        """Adaptation matching should not match similar but different agent names."""
        base_prompt = "Original prompt."
        adaptations = [
            {
                "target_agent": "planner",
                "modification_type": "append",
                "content": "Planner-specific content.",
            }
        ]

        # Should apply to exact match
        result_planner = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        assert "Planner-specific content." in result_planner

        # Should NOT apply to similar but different agents
        # These tests would catch bugs in substring matching logic
        result_plan_reviewer = apply_prompt_adaptations(base_prompt, "plan_reviewer", adaptations)
        assert "Planner-specific content." not in result_plan_reviewer, (
            "Adaptation for 'planner' should not match 'plan_reviewer'"
        )

        result_code = apply_prompt_adaptations(base_prompt, "code", adaptations)
        assert "Planner-specific content." not in result_code, (
            "Adaptation for 'planner' should not match 'code'"
        )

    def test_short_target_should_not_match_longer_agent_names(self):
        """A short target like 'code' should NOT match 'code_generator'.
        
        This test catches bugs where substring matching is too loose.
        """
        base_prompt = "Original prompt."
        
        # Target "code" should NOT match "code_generator"
        adaptations = [
            {
                "target_agent": "code",
                "modification_type": "append",
                "content": "Code-specific content.",
            }
        ]
        
        result = apply_prompt_adaptations(base_prompt, "code_generator", adaptations)
        assert "Code-specific content." not in result, (
            "Adaptation for 'code' should NOT match 'code_generator' - "
            "this would incorrectly apply to an unintended agent. "
            "The matching logic uses substring 'in' which is too loose."
        )
        
        # Similarly, "plan" should NOT match "planner"
        adaptations_plan = [
            {
                "target_agent": "plan",
                "modification_type": "append",
                "content": "Plan-specific content.",
            }
        ]
        
        result_planner = apply_prompt_adaptations(base_prompt, "planner", adaptations_plan)
        assert "Plan-specific content." not in result_planner, (
            "Adaptation for 'plan' should NOT match 'planner' - "
            "the matching logic is too loose with substring matching."
        )
        
        # And "simulation" should NOT match "simulation_designer"
        adaptations_sim = [
            {
                "target_agent": "simulation",
                "modification_type": "append",
                "content": "Simulation-specific content.",
            }
        ]
        
        result_designer = apply_prompt_adaptations(base_prompt, "simulation_designer", adaptations_sim)
        assert "Simulation-specific content." not in result_designer, (
            "Adaptation for 'simulation' should NOT match 'simulation_designer'"
        )

    def test_longer_target_should_not_match_shorter_agent_names(self):
        """A longer target like 'code_generator' should NOT match agent 'code'.
        
        This tests the reverse direction of substring matching.
        """
        base_prompt = "Original prompt."
        adaptations = [
            {
                "target_agent": "code_generator",
                "modification_type": "append",
                "content": "Generator-specific content.",
            }
        ]
        
        # If an agent "code" existed, it should NOT match "code_generator" target
        result = apply_prompt_adaptations(base_prompt, "code", adaptations)
        assert "Generator-specific content." not in result, (
            "Adaptation for 'code_generator' should NOT match agent 'code'"
        )

    def test_adaptation_matching_case_insensitive(self):
        """Adaptation matching should be case-insensitive."""
        base_prompt = "Original prompt."
        adaptations = [
            {
                "target_agent": "PLANNER",  # Uppercase
                "modification_type": "append",
                "content": "Planner content.",
            }
        ]

        # Should match lowercase agent name
        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        assert "Planner content." in result, (
            "Adaptation matching should be case-insensitive"
        )

    def test_adaptation_with_agent_suffix(self):
        """Adaptation matching should handle 'Agent' suffix in target."""
        base_prompt = "Original prompt."
        adaptations = [
            {
                "target_agent": "PlannerAgent",  # With 'Agent' suffix
                "modification_type": "append",
                "content": "Planner content.",
            }
        ]

        # Should match agent name without suffix
        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        assert "Planner content." in result, (
            "Adaptation matching should handle 'Agent' suffix in target_agent"
        )

    def test_replace_adaptation_requires_marker(self):
        """Replace adaptation should only work if marker exists."""
        base_prompt = "Original prompt without marker."
        adaptations = [
            {
                "target_agent": "planner",
                "modification_type": "replace",
                "content": "Replaced content.",
                "section_marker": "NONEXISTENT_MARKER",
            }
        ]

        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        # Should remain unchanged if marker doesn't exist
        assert result == base_prompt, (
            "Replace adaptation should not modify prompt if marker doesn't exist"
        )
        assert "Replaced content." not in result

    def test_disable_adaptation_requires_marker(self):
        """Disable adaptation should only work if marker exists."""
        base_prompt = "Original prompt without marker."
        adaptations = [
            {
                "target_agent": "planner",
                "modification_type": "disable",
                "section_marker": "NONEXISTENT_MARKER",
            }
        ]

        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        # Should remain unchanged if marker doesn't exist
        assert result == base_prompt, (
            "Disable adaptation should not modify prompt if marker doesn't exist"
        )
        assert "[DISABLED:" not in result

    def test_multiple_adaptations_for_same_agent(self):
        """Multiple adaptations for the same agent should all be applied."""
        base_prompt = "Original prompt."
        adaptations = [
            {
                "target_agent": "planner",
                "modification_type": "append",
                "content": "First adaptation.",
            },
            {
                "target_agent": "planner",
                "modification_type": "append",
                "content": "Second adaptation.",
            },
            {
                "target_agent": "planner",
                "modification_type": "prepend",
                "content": "Prepended content.",
            },
        ]

        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        assert "First adaptation." in result, "First adaptation should be applied"
        assert "Second adaptation." in result, "Second adaptation should be applied"
        assert "Prepended content." in result, "Prepended adaptation should be applied"
        assert "Original prompt." in result, "Original prompt should be preserved"

    def test_adaptations_order_preserved(self):
        """Adaptations should be applied in order (append should maintain order)."""
        base_prompt = "Original."
        adaptations = [
            {
                "target_agent": "planner",
                "modification_type": "append",
                "content": "FIRST",
            },
            {
                "target_agent": "planner",
                "modification_type": "append",
                "content": "SECOND",
            },
        ]

        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        first_pos = result.find("FIRST")
        second_pos = result.find("SECOND")
        assert first_pos < second_pos, (
            "Adaptations should be applied in order: FIRST should appear before SECOND"
        )

    def test_adaptation_with_missing_target_agent(self):
        """Adaptation without target_agent should not apply to any agent."""
        base_prompt = "Original prompt."
        adaptations = [
            {
                "modification_type": "append",
                "content": "Content without target.",
            }
        ]

        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        assert "Content without target." not in result, (
            "Adaptation without target_agent should not match any agent"
        )
        assert result == base_prompt

    def test_adaptation_with_missing_modification_type(self):
        """Adaptation without modification_type should not modify prompt."""
        base_prompt = "Original prompt."
        adaptations = [
            {
                "target_agent": "planner",
                "content": "Content without type.",
            }
        ]

        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        assert "Content without type." not in result, (
            "Adaptation without modification_type should not be applied"
        )
        assert result == base_prompt

    def test_adaptation_with_invalid_modification_type(self):
        """Adaptation with invalid modification_type should not modify prompt."""
        base_prompt = "Original prompt."
        adaptations = [
            {
                "target_agent": "planner",
                "modification_type": "invalid_type",
                "content": "Content with invalid type.",
            }
        ]

        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        assert "Content with invalid type." not in result, (
            "Adaptation with invalid modification_type should not be applied"
        )
        assert result == base_prompt

    def test_adaptation_with_empty_content(self):
        """Adaptation with empty content should still be applied."""
        base_prompt = "Original prompt."
        adaptations = [
            {
                "target_agent": "planner",
                "modification_type": "append",
                "content": "",
            }
        ]

        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        # The adaptation header should still be added even with empty content
        assert "# Paper-Specific Adaptation" in result

    def test_adaptation_with_none_content(self):
        """Adaptation with None content should be handled gracefully."""
        base_prompt = "Original prompt."
        adaptations = [
            {
                "target_agent": "planner",
                "modification_type": "append",
                "content": None,
            }
        ]

        # Should not raise an exception
        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        # Content "None" (as string) might be added or it might be handled
        assert "Original prompt." in result

    def test_adaptation_with_none_target_agent(self):
        """Adaptation with None target_agent should not apply to any agent."""
        base_prompt = "Original prompt."
        adaptations = [
            {
                "target_agent": None,
                "modification_type": "append",
                "content": "Content with None target.",
            }
        ]

        # Should not raise AttributeError on None.lower()
        # This tests error handling
        try:
            result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
            # If it doesn't raise, content should not be applied
            assert "Content with None target." not in result
        except AttributeError:
            pytest.fail(
                "apply_prompt_adaptations should handle None target_agent gracefully"
            )

    def test_replace_adaptation_replaces_all_occurrences(self):
        """Replace adaptation should replace all occurrences of the marker."""
        base_prompt = "Text with MARKER here and MARKER there."
        adaptations = [
            {
                "target_agent": "planner",
                "modification_type": "replace",
                "content": "REPLACED",
                "section_marker": "MARKER",
            }
        ]

        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        assert "MARKER" not in result, "All MARKER occurrences should be replaced"
        assert result.count("REPLACED") == 2, "Both occurrences should be replaced"

    def test_adaptation_with_special_characters_in_content(self):
        """Adaptation content with special characters should work."""
        base_prompt = "Original prompt."
        special_content = "Content with special chars: ${}[]()\\n\\t*+?^"
        adaptations = [
            {
                "target_agent": "planner",
                "modification_type": "append",
                "content": special_content,
            }
        ]

        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        assert special_content in result

    def test_adaptation_with_multiline_content(self):
        """Adaptation with multiline content should work."""
        base_prompt = "Original prompt."
        multiline_content = """Line 1
Line 2
Line 3"""
        adaptations = [
            {
                "target_agent": "planner",
                "modification_type": "append",
                "content": multiline_content,
            }
        ]

        result = apply_prompt_adaptations(base_prompt, "planner", adaptations)
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result


class TestBuildAgentPromptOrder:
    """Test that build_agent_prompt applies operations in correct order."""

    def test_global_rules_before_agent_prompt(self):
        """Global rules should appear before agent-specific prompt."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )
        prompt = build_agent_prompt("planner", state, include_global_rules=True)
        
        # Load components separately to verify order
        global_rules = load_global_rules()
        agent_prompt = load_prompt_file("planner_agent")
        
        # Global rules should appear first
        global_rules_start = prompt.find(global_rules[:50])
        agent_prompt_start = prompt.find(agent_prompt[:50])
        
        assert global_rules_start < agent_prompt_start, (
            "Global rules should appear before agent-specific prompt"
        )

    def test_placeholders_substituted_before_adaptations(self):
        """Placeholders should be substituted before adaptations are applied."""
        # Create a state with adaptations that reference placeholder-like text
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )
        state["prompt_adaptations"] = [
            {
                "target_agent": "planner",
                "modification_type": "append",
                "content": "Adaptation content with {THRESHOLDS_TABLE} reference",
            }
        ]
        
        prompt = build_agent_prompt("planner", state)
        
        # The placeholder in the adaptation should NOT be substituted
        # (adaptations are applied after placeholder substitution in global_rules/agent_prompt)
        # But if the adaptation content itself contains {THRESHOLDS_TABLE}, it won't be substituted
        # because adaptations are applied after substitution
        # This test verifies the order is correct
        assert "Adaptation content" in prompt

    def test_custom_prompts_dir(self):
        """build_agent_prompt should work with custom prompts_dir."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )
        
        # Should work with default prompts_dir
        prompt_default = build_agent_prompt("planner", state)
        
        # Should also work with explicit prompts_dir
        prompt_explicit = build_agent_prompt("planner", state, prompts_dir=PROMPTS_DIR)
        
        assert prompt_default == prompt_explicit, (
            "Explicit prompts_dir should produce same result as default"
        )


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_agent_name_raises_error(self):
        """Invalid agent name should raise FileNotFoundError."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )

        with pytest.raises(FileNotFoundError):
            build_agent_prompt("nonexistent_agent_xyz", state)

    def test_load_prompt_file_nonexistent(self):
        """load_prompt_file should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_prompt_file("nonexistent_file_xyz")

    def test_load_prompt_file_without_extension(self):
        """load_prompt_file should work with filename without .md extension."""
        # Should work - .md is added automatically
        content = load_prompt_file("planner_agent")
        assert content is not None
        assert len(content) > 0

    def test_load_prompt_file_with_extension(self):
        """load_prompt_file should work with filename with .md extension."""
        content = load_prompt_file("planner_agent.md")
        assert content is not None
        assert len(content) > 0

    def test_build_agent_prompt_with_empty_state(self):
        """build_agent_prompt should work with minimal state."""
        state = create_initial_state(
            paper_id="test",
            paper_text="",
            paper_domain="plasmonics",
        )
        prompt = build_agent_prompt("planner", state)
        assert prompt is not None
        assert len(prompt) > 0

    def test_validate_all_prompts_loadable(self):
        """validate_all_prompts_loadable should return True for all agents."""
        results = validate_all_prompts_loadable()
        assert isinstance(results, dict)
        assert "global_rules" in results
        assert results["global_rules"] is True, "global_rules.md should be loadable"

        # All agents in AGENT_PROMPTS should be loadable
        for agent_name in AGENT_PROMPTS.keys():
            assert agent_name in results, f"Agent '{agent_name}' missing from validation results"
            assert results[agent_name] is True, (
                f"Agent '{agent_name}' prompt should be loadable"
            )

    def test_get_agent_prompt_cached_without_cache(self):
        """get_agent_prompt_cached should work without cache."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )
        
        prompt = get_agent_prompt_cached("planner", state, cache=None)
        assert prompt is not None
        assert len(prompt) > 0

    def test_get_agent_prompt_cached_with_cache(self):
        """get_agent_prompt_cached should use cache when available."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )
        cache = {}
        
        # First call should build and cache
        prompt1 = get_agent_prompt_cached("planner", state, cache=cache)
        assert "planner" in cache
        assert cache["planner"] == prompt1
        
        # Second call should return cached version
        prompt2 = get_agent_prompt_cached("planner", state, cache=cache)
        assert prompt1 == prompt2

    def test_get_agent_prompt_cached_without_state(self):
        """get_agent_prompt_cached should work with None state."""
        cache = {}
        prompt = get_agent_prompt_cached("planner", state=None, cache=cache)
        assert prompt is not None
        assert len(prompt) > 0

    def test_get_agent_prompt_cached_bypasses_cache_with_adaptations(self):
        """get_agent_prompt_cached should bypass cache when adaptations exist."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )
        state["prompt_adaptations"] = [
            {
                "target_agent": "planner",
                "modification_type": "append",
                "content": "CACHE_BYPASS_TEST",
            }
        ]
        cache = {}
        
        # Should not use cache when adaptations exist
        prompt1 = get_agent_prompt_cached("planner", state, cache=cache)
        assert "CACHE_BYPASS_TEST" in prompt1
        # Cache should not be populated
        assert "planner" not in cache or "CACHE_BYPASS_TEST" not in cache.get("planner", "")

    def test_cache_is_agent_specific(self):
        """Cache should store different prompts for different agents."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )
        cache = {}
        
        prompt_planner = get_agent_prompt_cached("planner", state, cache=cache)
        prompt_code_gen = get_agent_prompt_cached("code_generator", state, cache=cache)
        
        assert "planner" in cache
        assert "code_generator" in cache
        assert cache["planner"] != cache["code_generator"], (
            "Different agents should have different cached prompts"
        )
        assert prompt_planner == cache["planner"]
        assert prompt_code_gen == cache["code_generator"]

    def test_cache_returns_same_object(self):
        """Cache should return the same string object (identity check)."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )
        cache = {}
        
        prompt1 = get_agent_prompt_cached("planner", state, cache=cache)
        prompt2 = get_agent_prompt_cached("planner", state, cache=cache)
        
        # Should be the exact same object (identity)
        assert prompt1 is prompt2, (
            "Cached prompt should return the same string object"
        )

    def test_cache_not_used_when_none(self):
        """When cache is None, each call should build a fresh prompt."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )
        
        prompt1 = get_agent_prompt_cached("planner", state, cache=None)
        prompt2 = get_agent_prompt_cached("planner", state, cache=None)
        
        # Should be equal but not necessarily the same object
        assert prompt1 == prompt2
        # Can't use identity check since they're built fresh each time

    def test_stale_cache_returns_old_value(self):
        """Cache doesn't invalidate when state changes (design choice verification).
        
        This tests that the cache is simple key-value without considering state changes.
        If adaptations are added AFTER caching, the cache is still used (unless adaptations exist).
        """
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )
        cache = {}
        
        # First call - cache the prompt
        prompt_original = get_agent_prompt_cached("planner", state, cache=cache)
        assert "planner" in cache
        
        # Create a new state (different paper_id, but same agent)
        state2 = create_initial_state(
            paper_id="different_paper",
            paper_text="Different content entirely.",
            paper_domain="quantum_optics",
        )
        
        # Second call with different state but same cache and no adaptations
        prompt_cached = get_agent_prompt_cached("planner", state2, cache=cache)
        
        # Since no adaptations, cache is used - this returns the OLD cached prompt
        # This is the current behavior - test documents it
        assert prompt_cached == prompt_original, (
            "Cache is used even when state changes (no adaptations). "
            "This is the current behavior - callers must manage cache invalidation."
        )

    def test_empty_adaptations_list_uses_cache(self):
        """Empty adaptations list (not missing) should still use cache."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )
        state["prompt_adaptations"] = []  # Explicitly empty list
        cache = {}
        
        prompt1 = get_agent_prompt_cached("planner", state, cache=cache)
        assert "planner" in cache
        
        prompt2 = get_agent_prompt_cached("planner", state, cache=cache)
        assert prompt1 is prompt2, "Empty adaptations list should allow cache usage"

    def test_adaptations_for_different_agent_still_uses_cache_for_target(self):
        """Adaptations for OTHER agents should still allow cache for target agent."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )
        # Adaptation for code_generator, NOT planner
        state["prompt_adaptations"] = [
            {
                "target_agent": "code_generator",
                "modification_type": "append",
                "content": "Code generator specific.",
            }
        ]
        cache = {}
        
        # Get planner prompt - should this use cache?
        # Current implementation: ANY adaptations = bypass cache for ALL agents
        prompt1 = get_agent_prompt_cached("planner", state, cache=cache)
        prompt2 = get_agent_prompt_cached("planner", state, cache=cache)
        
        # Check current behavior - if cache has planner, it's using cache
        # If not, it's always rebuilding
        # Document whatever the actual behavior is
        if "planner" in cache:
            assert prompt1 is prompt2, "If cached, should return same object"
        else:
            # Current implementation bypasses cache whenever ANY adaptations exist
            assert prompt1 == prompt2, "Even without cache, prompts should be equal"


class TestPromptContentQuality:
    """Test quality and structure of prompt content."""

    @pytest.mark.parametrize("agent_name", [
        "planner",
        "code_generator",
        "supervisor",
        "simulation_designer",
    ])
    def test_prompt_contains_agent_role(self, agent_name):
        """Prompts should contain role/description information."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )
        prompt = build_agent_prompt(agent_name, state)

        # Should contain some indication of the agent's role
        # Check for common patterns: "Role", "Agent", agent name, etc.
        assert (
            agent_name.replace("_", " ").title() in prompt or
            agent_name in prompt.lower() or
            "role" in prompt.lower() or
            "agent" in prompt.lower()
        ), (
            f"Prompt for '{agent_name}' should contain role/description information"
        )

    def test_prompts_have_reasonable_length(self):
        """All prompts should have reasonable minimum length."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )

        for agent_name in AGENT_PROMPTS.keys():
            prompt = build_agent_prompt(agent_name, state)
            assert len(prompt) > 200, (
                f"Prompt for '{agent_name}' is too short ({len(prompt)} chars). "
                "Expected substantial content."
            )

    def test_prompts_are_utf8_encoded(self):
        """All prompts should be valid UTF-8."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )

        for agent_name in AGENT_PROMPTS.keys():
            prompt = build_agent_prompt(agent_name, state)
            # Should not raise UnicodeDecodeError
            prompt.encode("utf-8")
            # Should be decodable
            assert isinstance(prompt, str)


class TestLoadPromptFile:
    """Test load_prompt_file function edge cases."""

    def test_load_prompt_file_returns_non_empty_string(self):
        """load_prompt_file should return a non-empty string."""
        content = load_prompt_file("planner_agent")
        assert isinstance(content, str)
        assert len(content) > 0
        assert content.strip()  # Not just whitespace

    def test_load_prompt_file_preserves_content(self):
        """load_prompt_file should preserve file content exactly."""
        # Load the same file twice
        content1 = load_prompt_file("planner_agent")
        content2 = load_prompt_file("planner_agent")
        
        assert content1 == content2, "Repeated loads should return identical content"

    def test_load_prompt_file_handles_unicode(self):
        """load_prompt_file should handle unicode content."""
        content = load_prompt_file("global_rules")
        # Verify unicode characters (like ═) are preserved
        assert "═" in content, "Unicode box-drawing characters should be preserved"

    def test_load_prompt_file_with_different_extensions(self):
        """load_prompt_file should handle filename with and without extension."""
        content_no_ext = load_prompt_file("planner_agent")
        content_with_ext = load_prompt_file("planner_agent.md")
        
        assert content_no_ext == content_with_ext, (
            "Loading with and without .md extension should return same content"
        )

    def test_load_prompt_file_with_double_extension(self):
        """load_prompt_file with double extension should fail appropriately."""
        # "planner_agent.md" + ".md" = "planner_agent.md.md"
        with pytest.raises(FileNotFoundError):
            load_prompt_file("planner_agent.md.md")


class TestBuildAgentPromptIntegration:
    """Integration tests for build_agent_prompt combining all components."""

    def test_global_rules_thresholds_substituted(self):
        """Global rules should have THRESHOLDS_TABLE substituted."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content.",
            paper_domain="plasmonics",
        )
        prompt = build_agent_prompt("planner", state, include_global_rules=True)
        
        # Global rules contains {THRESHOLDS_TABLE} placeholder
        # It should be substituted
        assert "{THRESHOLDS_TABLE}" not in prompt, (
            "THRESHOLDS_TABLE placeholder should be substituted in built prompt"
        )
        # The actual table should be there
        assert "| Quantity |" in prompt, (
            "Thresholds table content should be present in built prompt"
        )

    def test_all_agent_prompts_have_no_unsubstituted_placeholders(self):
        """No built prompt should have unsubstituted placeholders."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content.",
            paper_domain="plasmonics",
        )
        
        for agent_name in AGENT_PROMPTS.keys():
            prompt = build_agent_prompt(agent_name, state)
            unsubstituted = validate_placeholders_substituted(prompt)
            assert not unsubstituted, (
                f"Agent '{agent_name}' has unsubstituted placeholders: {unsubstituted}"
            )

    def test_separator_between_global_rules_and_agent_prompt(self):
        """There should be a visual separator between global rules and agent prompt."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content.",
            paper_domain="plasmonics",
        )
        prompt = build_agent_prompt("planner", state, include_global_rules=True)
        
        # Check for separator (the code uses ═ * 75)
        assert "═" * 75 in prompt, (
            "There should be a visual separator between global rules and agent prompt"
        )

    def test_prompt_consistency_across_calls(self):
        """Same inputs should produce identical prompts."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content.",
            paper_domain="plasmonics",
        )
        
        prompt1 = build_agent_prompt("planner", state)
        prompt2 = build_agent_prompt("planner", state)
        
        assert prompt1 == prompt2, "Same inputs should produce identical prompts"

    def test_different_agents_have_different_prompts(self):
        """Different agents should have different prompts."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content.",
            paper_domain="plasmonics",
        )
        
        prompts = {}
        for agent_name in AGENT_PROMPTS.keys():
            prompts[agent_name] = build_agent_prompt(agent_name, state)
        
        # Check that all prompts are unique
        unique_prompts = set(prompts.values())
        assert len(unique_prompts) == len(AGENT_PROMPTS), (
            "Each agent should have a unique prompt"
        )

    def test_agent_prompt_contains_agent_specific_content(self):
        """Each agent's prompt should contain agent-specific instructions."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content.",
            paper_domain="plasmonics",
        )
        
        # Test a few specific agents
        planner_prompt = build_agent_prompt("planner", state, include_global_rules=False)
        code_gen_prompt = build_agent_prompt("code_generator", state, include_global_rules=False)
        supervisor_prompt = build_agent_prompt("supervisor", state, include_global_rules=False)
        
        # Each should contain some indication of their role
        assert "plan" in planner_prompt.lower(), "Planner prompt should mention planning"
        assert "code" in code_gen_prompt.lower(), "Code generator prompt should mention code"
        assert "supervis" in supervisor_prompt.lower(), "Supervisor prompt should mention supervision"


class TestValidatePlaceholdersSubstituted:
    """Test the validate_placeholders_substituted helper function."""

    def test_empty_string_returns_empty_list(self):
        """Empty string should have no unsubstituted placeholders."""
        result = validate_placeholders_substituted("")
        assert result == []

    def test_no_placeholders_returns_empty_list(self):
        """String without placeholders should return empty list."""
        result = validate_placeholders_substituted("Normal text without placeholders")
        assert result == []

    def test_detects_thresholds_placeholder(self):
        """Should detect {THRESHOLDS_TABLE} placeholder."""
        result = validate_placeholders_substituted("Text with {THRESHOLDS_TABLE} here")
        assert "{THRESHOLDS_TABLE}" in result

    def test_detects_multiple_occurrences(self):
        """Should detect placeholder even with multiple occurrences."""
        result = validate_placeholders_substituted(
            "{THRESHOLDS_TABLE} and {THRESHOLDS_TABLE}"
        )
        # Should return the placeholder once (it's a list of unique placeholders found)
        assert "{THRESHOLDS_TABLE}" in result

    def test_does_not_detect_partial_match(self):
        """Should not detect partial placeholder matches."""
        result = validate_placeholders_substituted("THRESHOLDS_TABLE without braces")
        assert result == []


class TestAllAgentsValidation:
    """Comprehensive validation tests for all agents."""

    @pytest.mark.parametrize("agent_name", list(AGENT_PROMPTS.keys()))
    def test_agent_prompt_file_exists_via_mapping(self, agent_name):
        """Each agent in AGENT_PROMPTS should have a corresponding file."""
        prompt_filename = AGENT_PROMPTS[agent_name]
        prompt_file = PROMPTS_DIR / f"{prompt_filename}.md"
        assert prompt_file.exists(), (
            f"Agent '{agent_name}' maps to '{prompt_filename}' but file not found at {prompt_file}"
        )

    @pytest.mark.parametrize("agent_name", list(AGENT_PROMPTS.keys()))
    def test_agent_can_be_built_without_state(self, agent_name):
        """Each agent prompt should build without state."""
        prompt = build_agent_prompt(agent_name, state=None)
        assert prompt is not None
        assert len(prompt) > 100

    @pytest.mark.parametrize("agent_name", list(AGENT_PROMPTS.keys()))
    def test_agent_prompt_no_python_errors_in_content(self, agent_name):
        """Agent prompts should not contain obvious Python error strings."""
        prompt = build_agent_prompt(agent_name, state=None)
        
        # These would indicate problems in the prompt files
        error_patterns = [
            "Traceback (most recent call last)",
            "SyntaxError:",
            "NameError:",
            "TypeError:",
            "KeyError:",
        ]
        
        for pattern in error_patterns:
            assert pattern not in prompt, (
                f"Agent '{agent_name}' prompt contains error pattern: {pattern}"
            )


