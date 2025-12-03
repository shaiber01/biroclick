"""Tests for LLM error helper utilities in src.agents.base."""

import pytest

from src.agents.base import (
    create_llm_error_auto_approve,
    create_llm_error_escalation,
    create_llm_error_fallback,
)

class TestCreateLlmErrorAutoApprove:
    """Tests for create_llm_error_auto_approve function."""

    def test_creates_approve_verdict(self):
        """Should create response with approve verdict."""
        error = Exception("API timeout")
        
        result = create_llm_error_auto_approve("code_reviewer", error)
        
        assert result["verdict"] == "approve"
        assert len(result["issues"]) == 1
        assert result["issues"][0]["severity"] == "minor"
        assert "API timeout" in result["issues"][0]["description"]
        # Verify summary format
        assert result["summary"] == "Code Reviewer auto-approved due to LLM unavailability"

    def test_creates_pass_verdict_for_validators(self):
        """Should create response with pass verdict for validators."""
        error = Exception("Connection error")
        
        result = create_llm_error_auto_approve("execution_validator", error, default_verdict="pass")
        
        assert result["verdict"] == "pass"
        assert result["summary"] == "Execution Validator auto-passed due to LLM unavailability"

    def test_truncates_long_error_message(self):
        """Should truncate long error messages."""
        long_error = Exception("A" * 500)
        
        result = create_llm_error_auto_approve("test_agent", long_error, error_truncate_len=50)
        
        description = result["issues"][0]["description"]
        assert len(description) < 100
        expected_msg = ("A" * 500)[:50]
        assert expected_msg in description

    def test_includes_summary(self):
        """Should include summary message."""
        error = Exception("Test error")
        
        result = create_llm_error_auto_approve("code_reviewer", error)
        
        assert "Code Reviewer" in result["summary"]
        assert "auto-approve" in result["summary"].lower()

    def test_handles_empty_error_message(self):
        """Should handle empty exception message."""
        error = Exception("")
        result = create_llm_error_auto_approve("agent", error)
        assert "LLM review unavailable: " in result["issues"][0]["description"]

class TestCreateLlmErrorEscalation:
    """Tests for create_llm_error_escalation function."""

    def test_creates_escalation_response(self):
        """Should create user escalation response."""
        error = Exception("API key invalid")
        
        result = create_llm_error_escalation("code_generator", "code_generation", error)
        
        assert result["workflow_phase"] == "code_generation"
        assert result["ask_user_trigger"] == "llm_error"
        assert result["awaiting_user_input"] is True
        assert len(result["pending_user_questions"]) == 1

    def test_includes_error_in_question(self):
        """Should include error message in question."""
        error = Exception("API key invalid")
        
        result = create_llm_error_escalation("planner", "planning", error)
        
        assert "API key invalid" in result["pending_user_questions"][0]

    def test_truncates_long_error_in_question(self):
        """Should truncate long error messages in question."""
        long_error = Exception("B" * 1000)
        
        result = create_llm_error_escalation("test", "test", long_error, error_truncate_len=100)
        
        assert len(result["pending_user_questions"][0]) < 200

    def test_formats_agent_name(self):
        """Should format agent name in question."""
        error = Exception("Error")
        
        result = create_llm_error_escalation("code_generator", "code_generation", error)
        
        assert "Code Generator" in result["pending_user_questions"][0]
        assert "Code Generator failed: Error. Please check API and try again." in result["pending_user_questions"][0]

class TestCreateLlmErrorFallback:
    """Tests for create_llm_error_fallback function."""

    def test_creates_fallback_handler(self):
        """Should create a callable fallback handler."""
        handler = create_llm_error_fallback("supervisor", "ok_continue")
        
        assert callable(handler)

    def test_handler_returns_verdict_and_feedback(self):
        """Should return dict with verdict and feedback."""
        handler = create_llm_error_fallback("supervisor", "ok_continue")
        error = Exception("Test error")
        
        result = handler(error)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "Test error" in result["supervisor_feedback"]

    def test_uses_custom_feedback_format(self):
        """Should use custom feedback format if provided."""
        handler = create_llm_error_fallback(
            "supervisor", "ok_continue", 
            feedback_msg="Custom message: {error}"
        )
        error = Exception("Connection lost")
        
        result = handler(error)
        
        assert result["supervisor_feedback"] == "Custom message: Connection lost"

    def test_truncates_error_in_feedback(self):
        """Should truncate long errors in feedback."""
        handler = create_llm_error_fallback("test", "default", error_truncate_len=20)
        error = Exception("C" * 100)
        
        result = handler(error)
        
        assert len(result["test_feedback"]) < 50

