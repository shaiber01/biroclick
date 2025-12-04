"""Tests for user_options module - single source of truth for user interactions."""

import pytest
from unittest.mock import patch, MagicMock

from src.agents.user_options import (
    UserOption,
    USER_OPTIONS,
    get_options_for_trigger,
    get_options_prompt,
    get_clarification_message,
    validate_no_collisions,
    match_option_by_keywords,
    match_user_response,
    extract_guidance_text,
    _TRIGGER_ALIASES,
)


class TestUserOption:
    """Tests for UserOption dataclass."""
    
    def test_keywords_includes_display(self):
        """Display should be included in keywords."""
        opt = UserOption(display="APPROVE_PLAN", description="desc", action="x")
        assert "APPROVE_PLAN" in opt.keywords
    
    def test_keywords_includes_aliases(self):
        """Aliases should be included in keywords."""
        opt = UserOption(
            display="APPROVE_PLAN", 
            description="desc", 
            action="x",
            aliases=["APPROVE", "ACCEPT"]
        )
        assert "APPROVE" in opt.keywords
        assert "ACCEPT" in opt.keywords
    
    def test_keywords_are_uppercase(self):
        """All keywords should be uppercase."""
        opt = UserOption(
            display="approve_plan", 
            description="desc", 
            action="x",
            aliases=["accept"]
        )
        assert all(kw.isupper() for kw in opt.keywords)
    
    def test_empty_aliases_default(self):
        """Empty aliases list by default."""
        opt = UserOption(display="TEST", description="desc", action="x")
        assert opt.aliases == []
        assert opt.keywords == ["TEST"]


class TestGetOptionsForTrigger:
    """Tests for get_options_for_trigger function."""
    
    def test_returns_options_for_known_trigger(self):
        """Should return options for known triggers."""
        options = get_options_for_trigger("replan_limit")
        assert len(options) >= 3
        displays = [opt.display for opt in options]
        assert "APPROVE_PLAN" in displays
        assert "GUIDANCE" in displays
        assert "STOP" in displays
    
    def test_returns_empty_for_unknown_trigger(self):
        """Should return empty list for unknown triggers."""
        options = get_options_for_trigger("nonexistent_trigger")
        assert options == []
    
    def test_resolves_trigger_aliases(self):
        """Should resolve trigger aliases to shared option sets."""
        # These three should all use "critical_error" options
        opt1 = get_options_for_trigger("missing_paper_text")
        opt2 = get_options_for_trigger("missing_stage_id")
        opt3 = get_options_for_trigger("progress_init_failed")
        
        # All should have same options (RETRY, STOP)
        displays1 = [o.display for o in opt1]
        displays2 = [o.display for o in opt2]
        displays3 = [o.display for o in opt3]
        
        assert displays1 == displays2 == displays3
        assert "RETRY" in displays1
        assert "STOP" in displays1


class TestGetOptionsPrompt:
    """Tests for get_options_prompt function."""
    
    def test_includes_all_options(self):
        """Prompt should include all option displays."""
        prompt = get_options_prompt("replan_limit")
        assert "APPROVE_PLAN" in prompt
        assert "GUIDANCE" in prompt
        assert "STOP" in prompt
    
    def test_includes_descriptions(self):
        """Prompt should include option descriptions."""
        prompt = get_options_prompt("replan_limit")
        # Check for part of the description
        assert "Force-accept" in prompt or "force-accept" in prompt.lower()
    
    def test_format_is_correct(self):
        """Prompt should have correct format."""
        prompt = get_options_prompt("replan_limit")
        assert prompt.startswith("Options:")
        assert "- APPROVE_PLAN:" in prompt
    
    def test_unknown_trigger_message(self):
        """Unknown trigger should return informative message."""
        prompt = get_options_prompt("nonexistent_trigger")
        assert "none defined" in prompt.lower()


class TestGetClarificationMessage:
    """Tests for get_clarification_message function."""
    
    def test_two_option_format(self):
        """Two options should use 'or' format."""
        msg = get_clarification_message("backtrack_approval")  # APPROVE, REJECT
        assert "APPROVE" in msg
        assert "REJECT" in msg
        assert "clarify" in msg.lower()
    
    def test_multiple_options_format(self):
        """Multiple options should use comma-separated format with 'or'."""
        msg = get_clarification_message("replan_limit")  # 3 options
        assert "APPROVE_PLAN" in msg
        assert "GUIDANCE" in msg
        assert "STOP" in msg
        assert "or" in msg.lower()
        assert "clarify" in msg.lower()
    
    def test_unknown_trigger_message(self):
        """Unknown trigger should return error message."""
        msg = get_clarification_message("nonexistent_trigger")
        assert "no valid options" in msg.lower() or "contact support" in msg.lower()


class TestValidateNoCollisions:
    """Tests for validate_no_collisions function."""
    
    def test_no_collisions_in_default_config(self):
        """Default configuration should have no collisions."""
        # Should not raise
        validate_no_collisions()
    
    def test_detects_collision(self):
        """Should detect keyword collisions within a trigger."""
        import src.agents.user_options as user_options_module
        
        original = user_options_module.USER_OPTIONS.copy()
        
        try:
            # Add a trigger with collision
            user_options_module.USER_OPTIONS["test_collision"] = [
                UserOption(display="OPT1", description="d", action="a", aliases=["SHARED"]),
                UserOption(display="OPT2", description="d", action="b", aliases=["SHARED"]),
            ]
            
            with pytest.raises(ValueError, match="collision"):
                validate_no_collisions()
        finally:
            # Restore - must clear and update to avoid lingering keys
            user_options_module.USER_OPTIONS.clear()
            user_options_module.USER_OPTIONS.update(original)


class TestMatchOptionByKeywords:
    """Tests for match_option_by_keywords function."""
    
    def test_matches_exact_display(self):
        """Should match exact display keyword."""
        opt = match_option_by_keywords("replan_limit", "APPROVE_PLAN")
        assert opt is not None
        assert opt.display == "APPROVE_PLAN"
    
    def test_matches_alias(self):
        """Should match alias keyword."""
        opt = match_option_by_keywords("replan_limit", "APPROVE")
        assert opt is not None
        assert opt.display == "APPROVE_PLAN"
    
    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        opt = match_option_by_keywords("replan_limit", "approve_plan")
        assert opt is not None
        assert opt.display == "APPROVE_PLAN"
    
    def test_matches_in_sentence(self):
        """Should match keyword within sentence."""
        opt = match_option_by_keywords("replan_limit", "I want to APPROVE the plan")
        assert opt is not None
        assert opt.display == "APPROVE_PLAN"
    
    def test_no_match_returns_none(self):
        """Should return None for unmatched response."""
        opt = match_option_by_keywords("replan_limit", "INVALID_OPTION")
        assert opt is None
    
    def test_empty_response_returns_none(self):
        """Should return None for empty response."""
        opt = match_option_by_keywords("replan_limit", "")
        assert opt is None
    
    def test_matches_first_option_on_multiple_matches(self):
        """When multiple options could match, should return first."""
        # "ACCEPT" is an alias for APPROVE_PLAN in replan_limit
        opt = match_option_by_keywords("replan_limit", "ACCEPT")
        assert opt is not None
        # Should match first option that has ACCEPT as alias
        assert opt.display == "APPROVE_PLAN"


class TestMatchUserResponse:
    """Tests for match_user_response function (hybrid matching)."""
    
    def test_matches_via_keywords(self):
        """Should match using keyword matching."""
        opt = match_user_response("replan_limit", "APPROVE_PLAN")
        assert opt is not None
        assert opt.display == "APPROVE_PLAN"
    
    def test_returns_none_for_no_match(self):
        """Should return None when no match found."""
        opt = match_user_response("replan_limit", "blah blah blah")
        assert opt is None
    
    @patch.dict("os.environ", {"REPROLAB_USE_LOCAL_LLM": "1"})
    def test_falls_back_to_llm(self):
        """Should fall back to LLM when keyword matching fails."""
        # Create mock ollama module
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {
            "message": {"content": "APPROVE_PLAN"}
        }
        
        # Patch the import inside classify_with_local_llm
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            # Use response that won't keyword match
            opt = match_user_response("replan_limit", "yes please go ahead")
            
            assert opt is not None
            assert opt.display == "APPROVE_PLAN"
            mock_ollama.chat.assert_called_once()
    
    @patch.dict("os.environ", {"REPROLAB_USE_LOCAL_LLM": "0"})
    def test_skips_llm_when_disabled(self):
        """Should not call LLM when disabled."""
        # This should not call LLM even with non-matching response
        opt = match_user_response("replan_limit", "yes please go ahead")
        assert opt is None


class TestExtractGuidanceText:
    """Tests for extract_guidance_text function."""
    
    def test_strips_guidance_prefix_with_colon(self):
        """Should strip 'GUIDANCE:' prefix."""
        text = extract_guidance_text("GUIDANCE: focus on resonance")
        assert text == "focus on resonance"
    
    def test_strips_guidance_prefix_without_colon(self):
        """Should strip 'GUIDANCE' prefix without colon."""
        text = extract_guidance_text("GUIDANCE focus on resonance")
        assert text == "focus on resonance"
    
    def test_strips_hint_prefix(self):
        """Should strip 'HINT' prefix."""
        text = extract_guidance_text("HINT: check boundary conditions")
        assert text == "check boundary conditions"
    
    def test_case_insensitive(self):
        """Should be case-insensitive."""
        text = extract_guidance_text("guidance: focus on resonance")
        assert text == "focus on resonance"
    
    def test_preserves_text_without_prefix(self):
        """Should preserve text without keyword prefix."""
        text = extract_guidance_text("just some text")
        assert text == "just some text"
    
    def test_custom_keywords(self):
        """Should support custom keyword list."""
        text = extract_guidance_text("CUSTOM: value", keywords=["CUSTOM"])
        assert text == "value"


class TestTriggerCoverage:
    """Tests to ensure all expected triggers are defined."""
    
    def test_all_handler_triggers_have_options(self):
        """All triggers used in trigger_handlers.py should have options defined."""
        # List of triggers from TRIGGER_HANDLERS in trigger_handlers.py
        expected_triggers = [
            "material_checkpoint",
            "code_review_limit",
            "design_review_limit",
            "execution_failure_limit",
            "physics_failure_limit",
            "analysis_limit",
            "context_overflow",
            "replan_limit",
            "backtrack_approval",
            "deadlock_detected",
            "llm_error",
            # "clarification" is special - free-form response
            "missing_paper_text",  # uses critical_error via alias
            "missing_stage_id",    # uses critical_error via alias
            "progress_init_failed", # uses critical_error via alias
            "supervisor_error",
            "missing_design",
            "no_stages_available",  # uses planning_error via alias
            "invalid_backtrack_target",  # uses planning_error via alias
            "backtrack_target_not_found",  # uses planning_error via alias
            "backtrack_limit",
            "invalid_backtrack_decision",
            "unknown_escalation",
        ]
        
        for trigger in expected_triggers:
            options = get_options_for_trigger(trigger)
            assert len(options) > 0, f"Trigger '{trigger}' has no options defined"
    
    def test_all_options_have_stop(self):
        """Most triggers should have a STOP option."""
        # Triggers that don't need STOP (special cases)
        no_stop_needed = {"backtrack_approval", "clarification"}
        
        for trigger in USER_OPTIONS:
            if trigger in no_stop_needed:
                continue
            
            options = get_options_for_trigger(trigger)
            displays = [opt.display for opt in options]
            actions = [opt.action for opt in options]
            
            # Either has STOP display or stop action
            has_stop = "STOP" in displays or "stop" in actions
            assert has_stop, f"Trigger '{trigger}' missing STOP option"


class TestOptionsConsistency:
    """Tests for option configuration consistency."""
    
    def test_all_options_have_required_fields(self):
        """All options should have display, description, and action."""
        for trigger, options in USER_OPTIONS.items():
            for opt in options:
                assert opt.display, f"Option in '{trigger}' missing display"
                assert opt.description, f"Option '{opt.display}' in '{trigger}' missing description"
                assert opt.action, f"Option '{opt.display}' in '{trigger}' missing action"
    
    def test_displays_are_uppercase(self):
        """All display values should be uppercase."""
        for trigger, options in USER_OPTIONS.items():
            for opt in options:
                assert opt.display == opt.display.upper(), \
                    f"Display '{opt.display}' in '{trigger}' should be uppercase"
    
    def test_actions_are_snake_case(self):
        """All action values should be lowercase snake_case."""
        for trigger, options in USER_OPTIONS.items():
            for opt in options:
                assert opt.action == opt.action.lower(), \
                    f"Action '{opt.action}' in '{trigger}' should be lowercase"
                assert " " not in opt.action, \
                    f"Action '{opt.action}' in '{trigger}' should not have spaces"

