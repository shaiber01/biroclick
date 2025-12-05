"""
Tests for ASK_USER Trigger Definitions Module.

Tests that all triggers are properly documented, that utility functions work correctly,
and that the schema maintains logical consistency between verdicts and actions.
"""

import pytest
from typing import Dict, Any

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
        "design_flaw_limit",  # From physics_check when design_flaw verdict hits limit
        "execution_failure_limit",
        "physics_failure_limit",
        "analysis_limit",
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
        # Error fallback triggers:
        "supervisor_error",
        "missing_design",
        "unknown_escalation",
        # Reviewer explicit escalation:
        "reviewer_escalation",
    }

    def test_ask_user_triggers_is_non_empty_dict(self):
        """Verify ASK_USER_TRIGGERS is a non-empty dictionary."""
        assert isinstance(ASK_USER_TRIGGERS, dict)
        assert len(ASK_USER_TRIGGERS) > 0, "ASK_USER_TRIGGERS should not be empty"

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

    def test_trigger_count_matches_expected(self):
        """Verify exact count of triggers matches expectations."""
        assert len(ASK_USER_TRIGGERS) == len(self.EXPECTED_TRIGGERS), (
            f"Expected {len(self.EXPECTED_TRIGGERS)} triggers, got {len(ASK_USER_TRIGGERS)}. "
            f"Expected: {sorted(self.EXPECTED_TRIGGERS)}, Got: {sorted(ASK_USER_TRIGGERS.keys())}"
        )

    def test_all_triggers_have_required_fields(self):
        """Test that all triggers have the required documentation fields."""
        for trigger_name, trigger_info in ASK_USER_TRIGGERS.items():
            assert isinstance(trigger_info, dict), (
                f"Trigger '{trigger_name}' info should be a dict, got {type(trigger_info)}"
            )
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
            assert len(description) >= 10, f"{trigger_name}: description too short ('{description}')"
            
            # Source Node
            source = trigger_info["source_node"]
            assert isinstance(source, str), f"{trigger_name}: source_node must be str"
            assert len(source) > 0, f"{trigger_name}: source_node is empty"
            
            # Expected Response Keys
            keys = trigger_info["expected_response_keys"]
            assert isinstance(keys, list), f"{trigger_name}: expected_response_keys must be list"
            assert len(keys) > 0, f"{trigger_name}: expected_response_keys is empty"
            assert all(isinstance(k, str) for k in keys), f"{trigger_name}: all response keys must be strings"
            assert all(len(k) > 0 for k in keys), f"{trigger_name}: response keys cannot be empty strings"
            # Check for duplicates
            assert len(keys) == len(set(keys)), f"{trigger_name}: has duplicate response keys"
            
            # Valid Verdicts (if present)
            verdicts = trigger_info.get("valid_verdicts")
            if verdicts is not None:
                assert isinstance(verdicts, list), f"{trigger_name}: valid_verdicts must be list or None"
                assert len(verdicts) > 0, f"{trigger_name}: valid_verdicts list is empty"
                assert all(isinstance(v, str) for v in verdicts), f"{trigger_name}: all verdicts must be strings"
                assert all(len(v) > 0 for v in verdicts), f"{trigger_name}: verdicts cannot be empty strings"
                # Enforce uppercase convention for verdicts
                assert all(v.isupper() for v in verdicts), f"{trigger_name}: verdicts should be uppercase (e.g. 'APPROVE')"
                # Check for duplicates
                assert len(verdicts) == len(set(verdicts)), f"{trigger_name}: has duplicate verdicts"

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

    def test_all_handlers_have_documented_triggers(self):
        """
        Verify that all handlers in TRIGGER_HANDLERS have corresponding documentation.
        Prevents handlers without proper documentation.
        """
        from src.agents.supervision.trigger_handlers import TRIGGER_HANDLERS
        
        for handler_trigger in TRIGGER_HANDLERS:
            assert handler_trigger in ASK_USER_TRIGGERS, (
                f"Handler '{handler_trigger}' exists in TRIGGER_HANDLERS but is not documented in ASK_USER_TRIGGERS"
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
                    # Verify the action description is meaningful
                    action_desc = action[verdict]
                    assert isinstance(action_desc, str), (
                        f"Trigger '{trigger_name}' action for '{verdict}' should be a string description"
                    )
                    assert len(action_desc) > 5, (
                        f"Trigger '{trigger_name}' action for '{verdict}' is too short: '{action_desc}'"
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
                assert len(action) > 5, (
                    f"Trigger '{trigger_name}' free-form action description is too short: '{action}'"
                )


# ═══════════════════════════════════════════════════════════════════════
# Trigger Content Tests - Verify specific trigger contracts
# ═══════════════════════════════════════════════════════════════════════

class TestAskUserTriggerContent:
    """Tests for specific content of critical triggers to ensure contract stability."""
    
    def test_material_checkpoint_contract(self):
        """Verify material_checkpoint trigger contract."""
        info = ASK_USER_TRIGGERS.get("material_checkpoint")
        assert info is not None, "material_checkpoint trigger must exist"
        
        # Check required keys
        assert "verdict" in info["expected_response_keys"], "material_checkpoint must expect 'verdict'"
        assert "notes" in info["expected_response_keys"], "material_checkpoint must expect 'notes'"
        
        # Check verdicts
        verdicts = info["valid_verdicts"]
        assert verdicts is not None, "material_checkpoint must have defined verdicts"
        assert "APPROVE" in verdicts, "material_checkpoint must support APPROVE"
        assert "CHANGE_MATERIAL" in verdicts, "material_checkpoint must support CHANGE_MATERIAL"
        assert "CHANGE_DATABASE" in verdicts, "material_checkpoint must support CHANGE_DATABASE"
        assert "NEED_HELP" in verdicts, "material_checkpoint must support NEED_HELP"
        assert "STOP" in verdicts, "material_checkpoint must support STOP for user abort"
        
        # Verify actions exist for each verdict
        actions = info["supervisor_action"]
        assert isinstance(actions, dict)
        for verdict in verdicts:
            assert verdict in actions, f"material_checkpoint missing action for {verdict}"
            assert len(actions[verdict]) > 10, f"material_checkpoint action for {verdict} is too short"
    
    def test_code_review_limit_contract(self):
        """Verify code_review_limit trigger contract."""
        info = ASK_USER_TRIGGERS.get("code_review_limit")
        assert info is not None, "code_review_limit trigger must exist"
        
        assert "action" in info["expected_response_keys"], "code_review_limit must expect 'action'"
        assert "hint" in info["expected_response_keys"], "code_review_limit must expect 'hint'"
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None, "code_review_limit must have defined verdicts"
        assert "PROVIDE_HINT" in verdicts, "code_review_limit must support PROVIDE_HINT"
        assert "STOP" in verdicts, "code_review_limit must support STOP"
        assert "SKIP_STAGE" in verdicts, "code_review_limit must support SKIP_STAGE"
        
    def test_design_review_limit_contract(self):
        """Verify design_review_limit trigger contract."""
        info = ASK_USER_TRIGGERS.get("design_review_limit")
        assert info is not None, "design_review_limit trigger must exist"
        
        assert "action" in info["expected_response_keys"]
        assert "hint" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "PROVIDE_HINT" in verdicts
        assert "STOP" in verdicts
        assert "SKIP_STAGE" in verdicts
        
    def test_execution_failure_limit_contract(self):
        """Verify execution_failure_limit trigger contract."""
        info = ASK_USER_TRIGGERS.get("execution_failure_limit")
        assert info is not None
        
        assert "action" in info["expected_response_keys"]
        assert "guidance" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "RETRY_WITH_GUIDANCE" in verdicts
        assert "SKIP_STAGE" in verdicts
        assert "STOP" in verdicts
        
    def test_physics_failure_limit_contract(self):
        """Verify physics_failure_limit trigger contract."""
        info = ASK_USER_TRIGGERS.get("physics_failure_limit")
        assert info is not None
        
        assert "action" in info["expected_response_keys"]
        assert "guidance" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "RETRY_WITH_GUIDANCE" in verdicts
        assert "ACCEPT_PARTIAL" in verdicts
        assert "SKIP_STAGE" in verdicts
        assert "STOP" in verdicts
        
    def test_context_overflow_contract(self):
        """Verify context_overflow trigger contract."""
        info = ASK_USER_TRIGGERS.get("context_overflow")
        assert info is not None
        
        assert "action" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "SUMMARIZE_FEEDBACK" in verdicts
        assert "TRUNCATE_PAPER" in verdicts
        assert "SKIP_STAGE" in verdicts
        assert "STOP" in verdicts
        
    def test_replan_limit_contract(self):
        """Verify replan_limit trigger contract."""
        info = ASK_USER_TRIGGERS.get("replan_limit")
        assert info is not None
        
        assert "action" in info["expected_response_keys"]
        assert "guidance" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "FORCE_ACCEPT" in verdicts
        assert "PROVIDE_GUIDANCE" in verdicts
        assert "STOP" in verdicts
        
    def test_backtrack_approval_contract(self):
        """Verify backtrack_approval trigger contract."""
        info = ASK_USER_TRIGGERS.get("backtrack_approval")
        assert info is not None
        
        assert "approve" in info["expected_response_keys"]
        assert "alternative" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "APPROVE_BACKTRACK" in verdicts
        assert "REJECT_BACKTRACK" in verdicts
        assert "ALTERNATIVE" in verdicts
        
    def test_deadlock_detected_contract(self):
        """Verify deadlock_detected trigger contract."""
        info = ASK_USER_TRIGGERS.get("deadlock_detected")
        assert info is not None
        
        assert "action" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "GENERATE_REPORT" in verdicts
        assert "REPLAN" in verdicts
        assert "STOP" in verdicts
        
    def test_llm_error_contract(self):
        """Verify llm_error trigger contract."""
        info = ASK_USER_TRIGGERS.get("llm_error")
        assert info is not None
        
        assert "action" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "RETRY" in verdicts
        assert "SKIP_STAGE" in verdicts
        assert "STOP" in verdicts
        
    def test_clarification_contract(self):
        """Verify clarification trigger contract (free-form)."""
        info = ASK_USER_TRIGGERS.get("clarification")
        assert info is not None, "clarification trigger must exist"
        
        assert "clarification" in info["expected_response_keys"]
        assert info["valid_verdicts"] is None, "clarification should be free-form (None verdicts)"
        assert isinstance(info["supervisor_action"], str), "clarification action should be string"
        assert len(info["supervisor_action"]) > 10, "clarification action description too short"
        
    def test_missing_paper_text_contract(self):
        """Verify missing_paper_text trigger contract."""
        info = ASK_USER_TRIGGERS.get("missing_paper_text")
        assert info is not None
        
        assert "action" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "RETRY" in verdicts
        assert "STOP" in verdicts
        
    def test_missing_stage_id_contract(self):
        """Verify missing_stage_id trigger contract."""
        info = ASK_USER_TRIGGERS.get("missing_stage_id")
        assert info is not None
        
        assert "action" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "RETRY" in verdicts
        assert "STOP" in verdicts
        
    def test_no_stages_available_contract(self):
        """Verify no_stages_available trigger contract."""
        info = ASK_USER_TRIGGERS.get("no_stages_available")
        assert info is not None
        
        assert "action" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "REPLAN" in verdicts
        assert "STOP" in verdicts
        
    def test_progress_init_failed_contract(self):
        """Verify progress_init_failed trigger contract."""
        info = ASK_USER_TRIGGERS.get("progress_init_failed")
        assert info is not None
        
        assert "action" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "RETRY" in verdicts
        assert "STOP" in verdicts
        
    def test_backtrack_limit_contract(self):
        """Verify backtrack_limit trigger contract."""
        info = ASK_USER_TRIGGERS.get("backtrack_limit")
        assert info is not None
        
        assert "action" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "STOP" in verdicts
        assert "FORCE_CONTINUE" in verdicts
        
    def test_invalid_backtrack_target_contract(self):
        """Verify invalid_backtrack_target trigger contract."""
        info = ASK_USER_TRIGGERS.get("invalid_backtrack_target")
        assert info is not None
        
        assert "action" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "STOP" in verdicts
        assert "REPLAN" in verdicts
        
    def test_backtrack_target_not_found_contract(self):
        """Verify backtrack_target_not_found trigger contract."""
        info = ASK_USER_TRIGGERS.get("backtrack_target_not_found")
        assert info is not None
        
        assert "action" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "STOP" in verdicts
        assert "REPLAN" in verdicts
        
    def test_invalid_backtrack_decision_contract(self):
        """Verify invalid_backtrack_decision trigger contract."""
        info = ASK_USER_TRIGGERS.get("invalid_backtrack_decision")
        assert info is not None
        
        assert "action" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "STOP" in verdicts
        assert "CONTINUE" in verdicts
        
    def test_unknown_trigger_contract(self):
        """Verify unknown trigger contract - the fallback handler."""
        info = ASK_USER_TRIGGERS.get("unknown")
        assert info is not None, "unknown trigger must exist as fallback"
        
        assert "action" in info["expected_response_keys"]
        
        verdicts = info["valid_verdicts"]
        assert verdicts is not None
        assert "CONTINUE" in verdicts
        assert "STOP" in verdicts


# ═══════════════════════════════════════════════════════════════════════
# Utility Function Tests
# ═══════════════════════════════════════════════════════════════════════

class TestUtilityFunctions:
    """Tests for utility functions with edge cases."""
    
    def test_get_trigger_info_returns_correct_type(self):
        """Verify get_ask_user_trigger_info returns a dict."""
        result = get_ask_user_trigger_info("material_checkpoint")
        assert isinstance(result, dict)
        
    def test_get_trigger_info_known_trigger(self):
        """Test get_ask_user_trigger_info with valid trigger."""
        result = get_ask_user_trigger_info("context_overflow")
        assert result == ASK_USER_TRIGGERS["context_overflow"]
        assert result is ASK_USER_TRIGGERS["context_overflow"]  # Same object, not copy
        
    def test_get_trigger_info_unknown_trigger_returns_unknown(self):
        """Test that unknown triggers return 'unknown' trigger info."""
        result = get_ask_user_trigger_info("nonexistent_trigger")
        assert result == ASK_USER_TRIGGERS["unknown"]
        assert result["description"] == ASK_USER_TRIGGERS["unknown"]["description"]
        
    def test_get_trigger_info_empty_string(self):
        """Test that empty string returns 'unknown' trigger info."""
        result = get_ask_user_trigger_info("")
        assert result == ASK_USER_TRIGGERS["unknown"]
        
    def test_get_trigger_info_whitespace_string(self):
        """Test that whitespace-only string returns 'unknown' trigger info."""
        # Note: The function doesn't strip whitespace, so this tests actual behavior
        result = get_ask_user_trigger_info("   ")
        assert result == ASK_USER_TRIGGERS["unknown"]
        
    def test_get_trigger_info_case_sensitivity(self):
        """Test that trigger names are case-sensitive."""
        # These should NOT match valid triggers
        result_upper = get_ask_user_trigger_info("MATERIAL_CHECKPOINT")
        result_mixed = get_ask_user_trigger_info("Material_Checkpoint")
        
        # Both should fall back to unknown
        assert result_upper == ASK_USER_TRIGGERS["unknown"]
        assert result_mixed == ASK_USER_TRIGGERS["unknown"]
        
    def test_get_trigger_info_with_special_chars(self):
        """Test trigger info with special characters in name."""
        result = get_ask_user_trigger_info("trigger@with#special$chars")
        assert result == ASK_USER_TRIGGERS["unknown"]
        
    def test_get_trigger_info_with_unicode(self):
        """Test trigger info with unicode characters."""
        result = get_ask_user_trigger_info("trigger_日本語")
        assert result == ASK_USER_TRIGGERS["unknown"]

    def test_get_valid_triggers_returns_list(self):
        """Test that get_valid_triggers returns a list."""
        triggers = get_valid_triggers()
        assert isinstance(triggers, list)
        
    def test_get_valid_triggers_completeness(self):
        """Test that get_valid_triggers returns the complete list."""
        triggers = get_valid_triggers()
        assert len(triggers) == len(ASK_USER_TRIGGERS)
        assert set(triggers) == set(ASK_USER_TRIGGERS.keys())
        
    def test_get_valid_triggers_all_strings(self):
        """Test that all returned triggers are non-empty strings."""
        triggers = get_valid_triggers()
        assert all(isinstance(t, str) for t in triggers)
        assert all(len(t) > 0 for t in triggers)
        
    def test_get_valid_triggers_no_duplicates(self):
        """Test that there are no duplicate triggers."""
        triggers = get_valid_triggers()
        assert len(triggers) == len(set(triggers))

    def test_get_valid_verdicts_for_known_trigger_with_verdicts(self):
        """Test get_valid_verdicts_for_trigger with a trigger that has verdicts."""
        verdicts = get_valid_verdicts_for_trigger("material_checkpoint")
        assert isinstance(verdicts, list)
        assert "APPROVE" in verdicts
        assert len(verdicts) == len(ASK_USER_TRIGGERS["material_checkpoint"]["valid_verdicts"])
        
    def test_get_valid_verdicts_for_freeform_trigger(self):
        """Test get_valid_verdicts_for_trigger with a free-form trigger."""
        verdicts = get_valid_verdicts_for_trigger("clarification")
        assert verdicts is None
        
    def test_get_valid_verdicts_for_unknown_trigger(self):
        """Test get_valid_verdicts_for_trigger with unknown trigger."""
        verdicts = get_valid_verdicts_for_trigger("nonexistent")
        # Should return the unknown trigger's verdicts
        assert verdicts == ASK_USER_TRIGGERS["unknown"]["valid_verdicts"]
        
    def test_get_valid_verdicts_for_empty_string(self):
        """Test get_valid_verdicts_for_trigger with empty string."""
        verdicts = get_valid_verdicts_for_trigger("")
        assert verdicts == ASK_USER_TRIGGERS["unknown"]["valid_verdicts"]
        
    def test_get_valid_verdicts_returns_same_object(self):
        """Test that get_valid_verdicts returns the actual list from the dict."""
        verdicts = get_valid_verdicts_for_trigger("code_review_limit")
        expected = ASK_USER_TRIGGERS["code_review_limit"]["valid_verdicts"]
        assert verdicts is expected  # Same object reference


# ═══════════════════════════════════════════════════════════════════════
# Handler Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestHandlerIntegration:
    """Tests verifying handlers are consistent with trigger documentation."""
    
    def test_handler_functions_are_callable(self):
        """Verify all handlers are callable functions."""
        from src.agents.supervision.trigger_handlers import TRIGGER_HANDLERS
        
        for trigger_name, handler in TRIGGER_HANDLERS.items():
            assert callable(handler), f"Handler for '{trigger_name}' is not callable"
            
    def test_handler_registry_is_dict(self):
        """Verify TRIGGER_HANDLERS is a non-empty dict."""
        from src.agents.supervision.trigger_handlers import TRIGGER_HANDLERS
        
        assert isinstance(TRIGGER_HANDLERS, dict)
        assert len(TRIGGER_HANDLERS) > 0
            
    def test_all_nonfreeform_triggers_have_stop_option(self):
        """
        Most triggers should have a STOP option for user to abort workflow.
        This is a usability requirement - users should always have an escape hatch.
        """
        # Triggers exempt from requiring STOP:
        # - clarification: Free-form response, no verdicts
        # - backtrack_approval: Has REJECT_BACKTRACK which cancels without stopping
        exempt = {"clarification", "backtrack_approval"}
        
        for trigger_name, trigger_info in ASK_USER_TRIGGERS.items():
            verdicts = trigger_info.get("valid_verdicts")
            if verdicts is not None and trigger_name not in exempt:
                assert "STOP" in verdicts, (
                    f"Trigger '{trigger_name}' should have STOP verdict for user abort capability"
                )
                
    def test_all_limit_triggers_have_skip_option(self):
        """
        All *_limit triggers should have some form of skip/skip_stage option.
        """
        limit_triggers = [t for t in ASK_USER_TRIGGERS if "_limit" in t]
        
        for trigger_name in limit_triggers:
            verdicts = ASK_USER_TRIGGERS[trigger_name].get("valid_verdicts", [])
            has_skip = any("SKIP" in v for v in verdicts) if verdicts else False
            has_force = any("FORCE" in v for v in verdicts) if verdicts else False
            assert has_skip or has_force, (
                f"Limit trigger '{trigger_name}' should have SKIP or FORCE option to bypass the limit"
            )


# ═══════════════════════════════════════════════════════════════════════
# Source Node Validation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSourceNodeValidation:
    """Tests for source_node field validation."""
    
    def test_source_nodes_are_meaningful(self):
        """Verify source_node values are meaningful identifiers."""
        for trigger_name, trigger_info in ASK_USER_TRIGGERS.items():
            source = trigger_info["source_node"]
            # Source should be alphanumeric with underscores, slashes, or spaces for "any"
            assert source, f"{trigger_name}: source_node is empty"
            # Should not be just whitespace
            assert source.strip() == source, f"{trigger_name}: source_node has leading/trailing whitespace"
            
    def test_known_source_nodes(self):
        """Verify common source nodes are used correctly."""
        expected_sources = {
            "material_checkpoint": "material_checkpoint",
            "code_review_limit": "code_review",
            "design_review_limit": "design_review",
            "execution_failure_limit": "execution_check",
            "physics_failure_limit": "physics_check",
            "replan_limit": "plan_review",
            "backtrack_approval": "supervisor",
        }
        
        for trigger, expected_source in expected_sources.items():
            actual_source = ASK_USER_TRIGGERS[trigger]["source_node"]
            assert actual_source == expected_source, (
                f"Trigger '{trigger}' expected source '{expected_source}', got '{actual_source}'"
            )


# ═══════════════════════════════════════════════════════════════════════
# Response Keys Validation Tests  
# ═══════════════════════════════════════════════════════════════════════

class TestResponseKeysValidation:
    """Tests for expected_response_keys validation."""
    
    def test_common_response_keys(self):
        """Verify common response key patterns."""
        # Triggers that should have 'action' key
        action_triggers = [
            "code_review_limit", "design_review_limit", "execution_failure_limit",
            "physics_failure_limit", "context_overflow", "replan_limit",
            "deadlock_detected", "llm_error", "missing_paper_text", "missing_stage_id",
            "no_stages_available", "progress_init_failed", "backtrack_limit",
            "invalid_backtrack_target", "backtrack_target_not_found",
            "invalid_backtrack_decision", "unknown"
        ]
        
        for trigger in action_triggers:
            keys = ASK_USER_TRIGGERS[trigger]["expected_response_keys"]
            assert "action" in keys, f"Trigger '{trigger}' should have 'action' response key"
            
    def test_hint_guidance_keys_on_limit_triggers(self):
        """Verify limit triggers have hint/guidance keys for user assistance."""
        hint_triggers = ["code_review_limit", "design_review_limit"]
        guidance_triggers = ["execution_failure_limit", "physics_failure_limit", "replan_limit"]
        
        for trigger in hint_triggers:
            keys = ASK_USER_TRIGGERS[trigger]["expected_response_keys"]
            assert "hint" in keys, f"Trigger '{trigger}' should have 'hint' response key"
            
        for trigger in guidance_triggers:
            keys = ASK_USER_TRIGGERS[trigger]["expected_response_keys"]
            assert "guidance" in keys, f"Trigger '{trigger}' should have 'guidance' response key"


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
            
    def test_prompts_triggers_have_all_keys(self):
        """Verify prompts re-export has all trigger keys."""
        from src.prompts import ASK_USER_TRIGGERS as prompts_triggers
        
        assert set(prompts_triggers.keys()) == set(ASK_USER_TRIGGERS.keys())


# ═══════════════════════════════════════════════════════════════════════
# Immutability / Safety Tests
# ═══════════════════════════════════════════════════════════════════════

class TestTriggerDataImmutability:
    """Tests to ensure trigger data safety patterns."""
    
    def test_trigger_keys_are_lowercase_snake_case(self):
        """Verify all trigger names follow lowercase_snake_case convention."""
        import re
        pattern = re.compile(r'^[a-z][a-z0-9_]*$')
        
        for trigger_name in ASK_USER_TRIGGERS:
            assert pattern.match(trigger_name), (
                f"Trigger '{trigger_name}' doesn't follow lowercase_snake_case convention"
            )
            
    def test_no_duplicate_descriptions(self):
        """Verify no two triggers have the exact same description."""
        descriptions = [info["description"] for info in ASK_USER_TRIGGERS.values()]
        assert len(descriptions) == len(set(descriptions)), (
            "Found duplicate trigger descriptions - each trigger should have unique description"
        )
        
    def test_verdict_naming_convention(self):
        """Verify all verdicts follow UPPER_SNAKE_CASE convention."""
        import re
        pattern = re.compile(r'^[A-Z][A-Z0-9_]*$')
        
        for trigger_name, trigger_info in ASK_USER_TRIGGERS.items():
            verdicts = trigger_info.get("valid_verdicts")
            if verdicts:
                for verdict in verdicts:
                    assert pattern.match(verdict), (
                        f"Trigger '{trigger_name}' verdict '{verdict}' doesn't follow UPPER_SNAKE_CASE"
                    )


# ═══════════════════════════════════════════════════════════════════════
# Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge case tests for robustness."""
    
    def test_triggers_dict_is_not_empty(self):
        """Ensure ASK_USER_TRIGGERS is never empty."""
        assert ASK_USER_TRIGGERS, "ASK_USER_TRIGGERS should not be empty"
        assert len(ASK_USER_TRIGGERS) >= 10, "Expected at least 10 triggers defined"
        
    def test_unknown_trigger_always_exists(self):
        """The 'unknown' trigger must always exist as a fallback."""
        assert "unknown" in ASK_USER_TRIGGERS, "'unknown' trigger must exist as fallback"
        
    def test_get_trigger_info_handles_none_gracefully(self):
        """Test behavior when None is passed (if function allows it)."""
        # This tests the actual behavior - if it raises, that's documented
        try:
            result = get_ask_user_trigger_info(None)  # type: ignore
            # If it doesn't raise, it should return unknown
            assert result == ASK_USER_TRIGGERS["unknown"]
        except (TypeError, AttributeError):
            # Function doesn't handle None - that's acceptable but should be documented
            pass
            
    def test_verdict_lists_are_not_shared(self):
        """Ensure verdict lists are independent (no shared mutable objects)."""
        # Get two different triggers
        mat_verdicts = ASK_USER_TRIGGERS["material_checkpoint"]["valid_verdicts"]
        code_verdicts = ASK_USER_TRIGGERS["code_review_limit"]["valid_verdicts"]
        
        # They should not be the same list object
        assert mat_verdicts is not code_verdicts
        
    def test_all_triggers_have_valid_verdicts_key(self):
        """All triggers should have valid_verdicts key (even if None)."""
        for trigger_name, trigger_info in ASK_USER_TRIGGERS.items():
            assert "valid_verdicts" in trigger_info, (
                f"Trigger '{trigger_name}' missing 'valid_verdicts' key"
            )
