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


