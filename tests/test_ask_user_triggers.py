"""
Tests for ASK_USER Trigger Definitions Module.

Tests that all triggers are properly documented, that utility functions work correctly,
and that the schema maintains logical consistency between verdicts and actions.
"""

import pytest

from schemas.ask_user_triggers import (
    ASK_USER_TRIGGERS,
    get_ask_user_trigger_info,
    get_valid_triggers,
    get_valid_verdicts_for_trigger,
)


# ═══════════════════════════════════════════════════════════════════════
# Trigger Definition & Structure Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAskUserTriggersStructure:
    """Tests for ASK_USER_TRIGGERS dictionary structure and consistency."""
    
    REQUIRED_FIELDS = ["description", "source_node", "expected_response_keys", "supervisor_action"]
    
    # Define the exact set of triggers we expect to strictly enforce the contract.
    # This prevents accidental deletion or addition of triggers without updating tests.
    EXPECTED_TRIGGERS = {
        "material_checkpoint",
        "code_review_limit",
        "design_review_limit",
        "execution_failure_limit",
        "physics_failure_limit",
        "context_overflow",
        "replan_limit",
        "backtrack_approval",
        "clarification",
        "unknown",
        # Triggers found in codebase usage:
        "deadlock_detected",
        "llm_error",
        "invalid_backtrack_decision",
        "invalid_backtrack_target",
        "backtrack_target_not_found",
        "backtrack_limit",
        "missing_stage_id",
        "no_stages_available",
        "progress_init_failed",
        "missing_paper_text",
    }

    def test_all_expected_triggers_exist(self):
        """
        Verify that all defined expected triggers are present in the schema.
        This ensures no critical trigger is accidentally removed.
        """
        current_triggers = set(ASK_USER_TRIGGERS.keys())
        missing = self.EXPECTED_TRIGGERS - current_triggers
        assert not missing, f"Missing expected triggers: {missing}"
        
        # Verify no undocumented triggers exist
        extra = current_triggers - self.EXPECTED_TRIGGERS
        assert not extra, f"Found undocumented/untracked triggers: {extra}. Please update the test expectation if these are intentional."

    def test_all_triggers_have_required_fields(self):
        """Test that all triggers have the required documentation fields."""
        for trigger_name, trigger_info in ASK_USER_TRIGGERS.items():
            for field in self.REQUIRED_FIELDS:
                assert field in trigger_info, (
                    f"Trigger '{trigger_name}' missing required field '{field}'"
                )

    def test_field_types_and_content(self):
        """
        Verify types and non-empty content for all fields.
        This ensures the schema is not just present but valid.
        """
        for trigger_name, trigger_info in ASK_USER_TRIGGERS.items():
            # Description
            description = trigger_info["description"]
            assert isinstance(description, str), f"{trigger_name}: description must be str"
            assert len(description) > 0, f"{trigger_name}: description is empty"
            
            # Source Node
            source = trigger_info["source_node"]
            assert isinstance(source, str), f"{trigger_name}: source_node must be str"
            assert len(source) > 0, f"{trigger_name}: source_node is empty"
            
            # Expected Response Keys
            keys = trigger_info["expected_response_keys"]
            assert isinstance(keys, list), f"{trigger_name}: expected_response_keys must be list"
            assert len(keys) > 0, f"{trigger_name}: expected_response_keys is empty"
            assert all(isinstance(k, str) for k in keys), f"{trigger_name}: all response keys must be strings"
            
            # Valid Verdicts (if present)
            verdicts = trigger_info.get("valid_verdicts")
            if verdicts is not None:
                assert isinstance(verdicts, list), f"{trigger_name}: valid_verdicts must be list or None"
                assert len(verdicts) > 0, f"{trigger_name}: valid_verdicts list is empty"
                assert all(isinstance(v, str) for v in verdicts), f"{trigger_name}: all verdicts must be strings"
                # Enforce uppercase convention for verdicts
                assert all(v.isupper() for v in verdicts), f"{trigger_name}: verdicts should be uppercase (e.g. 'APPROVE')"

    def test_documented_triggers_have_handlers(self):
        """
        Verify that all documented triggers have a corresponding handler implementation.
        Exceptions: 'unknown' (handled by default fallback).
        """
        from src.agents.supervision.trigger_handlers import TRIGGER_HANDLERS
        
        # Triggers that are handled by default logic or special cases
        exempt = {"unknown"} 
        
        for trigger in ASK_USER_TRIGGERS:
            if trigger not in exempt:
                assert trigger in TRIGGER_HANDLERS, (
                    f"Trigger '{trigger}' is documented in schema but missing from TRIGGER_HANDLERS. "
                    "Logic will fall back to generic 'unknown' handling."
                )


    def test_consistency_verdicts_and_actions(self):
        """
        Strong Consistency Check:
        If valid_verdicts is a list, supervisor_action MUST be a dict,
        and it MUST contain an action for every verdict.
        """
        for trigger_name, trigger_info in ASK_USER_TRIGGERS.items():
            verdicts = trigger_info.get("valid_verdicts")
            action = trigger_info.get("supervisor_action")
            
            if verdicts is not None:
                # If we have specific verdicts, action must be a map handling each one
                assert isinstance(action, dict), (
                    f"Trigger '{trigger_name}' has verdicts but supervisor_action is not a dict mapping them"
                )
                
                # Check that every verdict has a corresponding action
                for verdict in verdicts:
                    assert verdict in action, (
                        f"Trigger '{trigger_name}' has verdict '{verdict}' but no defined supervisor_action for it"
                    )
                    
                # Check that action doesn't have orphan keys (actions for verdicts that don't exist)
                for action_key in action.keys():
                    assert action_key in verdicts, (
                        f"Trigger '{trigger_name}' has action key '{action_key}' which is not in valid_verdicts"
                    )

    def test_consistency_freeform_actions(self):
        """
        Strong Consistency Check:
        If valid_verdicts is None (free-form), supervisor_action MUST be a description string.
        """
        for trigger_name, trigger_info in ASK_USER_TRIGGERS.items():
            verdicts = trigger_info.get("valid_verdicts")
            action = trigger_info.get("supervisor_action")
            
            if verdicts is None:
                assert isinstance(action, str), (
                    f"Trigger '{trigger_name}' is free-form (verdicts=None) so supervisor_action should be a string description"
                )


class TestAskUserTriggerContent:
    """Tests for specific content of critical triggers to ensure contract stability."""
    
    def test_material_checkpoint_contract(self):
        """Verify material_checkpoint trigger contract."""
        info = ASK_USER_TRIGGERS.get("material_checkpoint")
        assert info is not None
        
        # Check keys
        assert "verdict" in info["expected_response_keys"]
        assert "notes" in info["expected_response_keys"]
        
        # Check verdicts
        verdicts = info["valid_verdicts"]
        assert "APPROVE" in verdicts
        assert "CHANGE_MATERIAL" in verdicts
        assert "CHANGE_DATABASE" in verdicts
        assert "NEED_HELP" in verdicts
    
    def test_code_review_limit_contract(self):
        """Verify code_review_limit trigger contract."""
        info = ASK_USER_TRIGGERS.get("code_review_limit")
        assert info is not None
        
        assert "action" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert "PROVIDE_HINT" in verdicts
        assert "STOP" in verdicts
        assert "SKIP_STAGE" in verdicts
        
    def test_clarification_contract(self):
        """Verify clarification trigger contract (free-form)."""
        info = ASK_USER_TRIGGERS.get("clarification")
        assert info is not None
        
        assert "clarification" in info["expected_response_keys"]
        assert info["valid_verdicts"] is None
        assert isinstance(info["supervisor_action"], str)


# ═══════════════════════════════════════════════════════════════════════
# Utility Function Tests
# ═══════════════════════════════════════════════════════════════════════

class TestUtilityFunctions:
    """Tests for utility functions with edge cases."""
    
    def test_get_trigger_info_edge_cases(self):
        """Test get_ask_user_trigger_info with various inputs."""
        # Unknown trigger
        assert get_ask_user_trigger_info("nonexistent_trigger") == ASK_USER_TRIGGERS["unknown"]
        # Empty string
        assert get_ask_user_trigger_info("") == ASK_USER_TRIGGERS["unknown"]
        # Valid trigger
        assert get_ask_user_trigger_info("context_overflow") == ASK_USER_TRIGGERS["context_overflow"]

    def test_get_valid_triggers_completeness(self):
        """Test that get_valid_triggers returns the complete list."""
        triggers = get_valid_triggers()
        assert isinstance(triggers, list)
        assert len(triggers) == len(ASK_USER_TRIGGERS)
        assert set(triggers) == set(ASK_USER_TRIGGERS.keys())

    def test_get_valid_verdicts_edge_cases(self):
        """Test get_valid_verdicts_for_trigger with various inputs."""
        # Non-existent trigger -> returns unknown trigger's verdicts
        assert get_valid_verdicts_for_trigger("nonexistent") == ASK_USER_TRIGGERS["unknown"]["valid_verdicts"]
        
        # Free-form trigger -> returns None
        assert get_valid_verdicts_for_trigger("clarification") is None
        
        # Standard trigger
        verdicts = get_valid_verdicts_for_trigger("material_checkpoint")
        assert isinstance(verdicts, list)
        assert "APPROVE" in verdicts


# ═══════════════════════════════════════════════════════════════════════
# Backward Compatibility Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Tests that imports from prompts.py still work (legacy support)."""
    
    def test_import_from_prompts_works(self):
        """Test that ASK_USER_TRIGGERS can still be imported from prompts."""
        try:
            from src.prompts import ASK_USER_TRIGGERS as prompts_triggers
            from src.prompts import get_ask_user_trigger_info as prompts_get_info
            
            # Should be the same object identity
            assert prompts_triggers is ASK_USER_TRIGGERS
            assert prompts_get_info is get_ask_user_trigger_info
        except ImportError:
            # If the project has moved away from this re-export, this test might fail
            # or can be removed. For now, we assert it works as per existing codebase.
            pytest.fail("Could not import ASK_USER_TRIGGERS from src.prompts")
