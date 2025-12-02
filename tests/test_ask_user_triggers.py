"""
Tests for ASK_USER Trigger Definitions Module.

Tests that all triggers are properly documented and that utility
functions work correctly.
"""

import pytest

from schemas.ask_user_triggers import (
    ASK_USER_TRIGGERS,
    get_ask_user_trigger_info,
    get_valid_triggers,
    get_valid_verdicts_for_trigger,
)


# ═══════════════════════════════════════════════════════════════════════
# Trigger Definition Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAskUserTriggers:
    """Tests for ASK_USER_TRIGGERS dictionary structure."""
    
    REQUIRED_FIELDS = ["description", "source_node", "expected_response_keys", "supervisor_action"]
    
    def test_all_triggers_have_required_fields(self):
        """Test that all triggers have the required documentation fields."""
        for trigger_name, trigger_info in ASK_USER_TRIGGERS.items():
            for field in self.REQUIRED_FIELDS:
                assert field in trigger_info, (
                    f"Trigger '{trigger_name}' missing required field '{field}'"
                )
    
    def test_all_triggers_have_description(self):
        """Test that all triggers have a non-empty description."""
        for trigger_name, trigger_info in ASK_USER_TRIGGERS.items():
            description = trigger_info.get("description", "")
            assert len(description) > 0, (
                f"Trigger '{trigger_name}' has empty description"
            )
    
    def test_all_triggers_have_source_node(self):
        """Test that all triggers specify a source node."""
        for trigger_name, trigger_info in ASK_USER_TRIGGERS.items():
            source = trigger_info.get("source_node", "")
            assert len(source) > 0, (
                f"Trigger '{trigger_name}' has empty source_node"
            )
    
    def test_all_triggers_have_response_keys(self):
        """Test that all triggers specify expected response keys."""
        for trigger_name, trigger_info in ASK_USER_TRIGGERS.items():
            keys = trigger_info.get("expected_response_keys", [])
            assert isinstance(keys, list), (
                f"Trigger '{trigger_name}' expected_response_keys should be a list"
            )
            assert len(keys) > 0, (
                f"Trigger '{trigger_name}' has empty expected_response_keys"
            )
    
    def test_valid_verdicts_is_list_or_none(self):
        """Test that valid_verdicts is either a list of strings or None."""
        for trigger_name, trigger_info in ASK_USER_TRIGGERS.items():
            verdicts = trigger_info.get("valid_verdicts")
            if verdicts is not None:
                assert isinstance(verdicts, list), (
                    f"Trigger '{trigger_name}' valid_verdicts should be list or None"
                )
                for verdict in verdicts:
                    assert isinstance(verdict, str), (
                        f"Trigger '{trigger_name}' verdict '{verdict}' should be string"
                    )
    
    def test_supervisor_action_is_documented(self):
        """Test that supervisor_action is either a string or dict mapping verdicts to actions."""
        for trigger_name, trigger_info in ASK_USER_TRIGGERS.items():
            action = trigger_info.get("supervisor_action")
            assert action is not None, (
                f"Trigger '{trigger_name}' missing supervisor_action"
            )
            assert isinstance(action, (str, dict)), (
                f"Trigger '{trigger_name}' supervisor_action should be str or dict"
            )
    
    def test_unknown_trigger_exists(self):
        """Test that 'unknown' trigger exists for fallback handling."""
        assert "unknown" in ASK_USER_TRIGGERS
        unknown_info = ASK_USER_TRIGGERS["unknown"]
        assert unknown_info.get("description") is not None


# ═══════════════════════════════════════════════════════════════════════
# Utility Function Tests
# ═══════════════════════════════════════════════════════════════════════

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_get_trigger_info_returns_known_trigger(self):
        """Test that get_ask_user_trigger_info returns info for known triggers."""
        info = get_ask_user_trigger_info("material_checkpoint")
        assert info["description"] == "Mandatory Stage 0 material validation requires user confirmation"
        assert info["source_node"] == "material_checkpoint"
    
    def test_get_trigger_info_returns_unknown_for_invalid(self):
        """Test that invalid triggers return 'unknown' trigger info."""
        info = get_ask_user_trigger_info("nonexistent_trigger")
        assert info == ASK_USER_TRIGGERS["unknown"]
    
    def test_get_valid_triggers_returns_all_keys(self):
        """Test that get_valid_triggers returns all trigger names."""
        triggers = get_valid_triggers()
        assert isinstance(triggers, list)
        assert "material_checkpoint" in triggers
        assert "code_review_limit" in triggers
        assert "unknown" in triggers
        assert len(triggers) == len(ASK_USER_TRIGGERS)
    
    def test_get_valid_verdicts_for_known_trigger(self):
        """Test that valid verdicts are returned for known triggers."""
        verdicts = get_valid_verdicts_for_trigger("material_checkpoint")
        assert verdicts is not None
        assert "APPROVE" in verdicts
        assert "CHANGE_DATABASE" in verdicts
    
    def test_get_valid_verdicts_for_clarification_returns_none(self):
        """Test that clarification trigger (free-form) returns None for verdicts."""
        verdicts = get_valid_verdicts_for_trigger("clarification")
        assert verdicts is None
    
    def test_get_valid_verdicts_for_unknown_trigger(self):
        """Test that unknown triggers return the 'unknown' verdicts."""
        verdicts = get_valid_verdicts_for_trigger("nonexistent")
        # Should return "unknown" trigger's verdicts
        assert verdicts == ASK_USER_TRIGGERS["unknown"]["valid_verdicts"]


# ═══════════════════════════════════════════════════════════════════════
# Backward Compatibility Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Tests that imports from prompts.py still work."""
    
    def test_import_from_prompts_works(self):
        """Test that ASK_USER_TRIGGERS can still be imported from prompts."""
        from src.prompts import ASK_USER_TRIGGERS as prompts_triggers
        from src.prompts import get_ask_user_trigger_info as prompts_get_info
        
        # Should be the same object
        assert prompts_triggers is ASK_USER_TRIGGERS
        assert prompts_get_info is get_ask_user_trigger_info
    
    def test_prompts_import_has_all_triggers(self):
        """Test that prompts import has all expected triggers."""
        from src.prompts import ASK_USER_TRIGGERS as prompts_triggers
        
        assert "material_checkpoint" in prompts_triggers
        assert "code_review_limit" in prompts_triggers
        assert "design_review_limit" in prompts_triggers



