"""Tests for with_context_check decorator."""

import pytest
from unittest.mock import patch

from src.agents.base import with_context_check

class TestWithContextCheck:
    """Tests for with_context_check decorator."""

    @patch("src.agents.base.check_context_or_escalate")
    def test_returns_escalation_when_awaiting_input(self, mock_check):
        """Should return escalation when context check requires user input."""
        mock_check.return_value = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
        }
        
        @with_context_check("test_node")
        def test_node(state):
            return {"result": "success"}
        
        result = test_node({"some": "state"})
        
        assert result["awaiting_user_input"] is True
        assert result["pending_user_questions"] == ["Context overflow"]

    @patch("src.agents.base.check_context_or_escalate")
    def test_updates_state_before_calling_func(self, mock_check):
        """Should update state with context changes BEFORE calling function."""
        mock_check.return_value = {"metrics": {"tokens": 100}}
        
        @with_context_check("test_node")
        def test_node(state):
            # Verify state has been updated inside function
            if state.get("metrics") != {"tokens": 100}:
                raise ValueError("State was not updated before function call")
            return {"result": "success"}
        
        # This should NOT raise ValueError
        result = test_node({"original": "data"})
        assert result["result"] == "success"

    @patch("src.agents.base.check_context_or_escalate")
    def test_continues_when_no_escalation(self, mock_check):
        """Should continue to wrapped function when no escalation needed."""
        mock_check.return_value = None
        
        @with_context_check("test_node")
        def test_node(state):
            return {"workflow_phase": "testing", "data": state.get("data")}
        
        result = test_node({"data": "test_value"})
        
        assert result["workflow_phase"] == "testing"
        assert result["data"] == "test_value"

    @patch("src.agents.base.check_context_or_escalate")
    def test_calls_check_with_correct_node_name(self, mock_check):
        """Should call check_context_or_escalate with correct node name."""
        mock_check.return_value = None
        
        @with_context_check("my_custom_node")
        def my_node(state):
            return {}
        
        my_node({"state": "data"})
        
        mock_check.assert_called_once()
        call_args = mock_check.call_args
        assert call_args[0][1] == "my_custom_node"

    @patch("src.agents.base.check_context_or_escalate")
    def test_preserves_function_metadata(self, mock_check):
        """Should preserve function name and docstring."""
        mock_check.return_value = None
        
        @with_context_check("test")
        def my_documented_function(state):
            """This is my docstring."""
            return {}
        
        assert my_documented_function.__name__ == "my_documented_function"
        assert "docstring" in my_documented_function.__doc__

    @patch("src.agents.base.check_context_or_escalate")
    def test_supports_kwargs_passing(self, mock_check):
        """Should support passing additional arguments to the decorated function."""
        # This test currently FAILS because the wrapper signature is fixed to (state)
        mock_check.return_value = None
        
        @with_context_check("test_kwargs")
        def node_with_args(state, extra_arg=None):
            return {"extra": extra_arg}
        
        result = node_with_args({}, extra_arg="working")
        assert result["extra"] == "working"

    @patch("src.agents.base.check_context_or_escalate")
    def test_propagates_exceptions(self, mock_check):
        """Should propagate exceptions from the decorated function."""
        mock_check.return_value = None
        
        @with_context_check("test_error")
        def error_node(state):
            raise ValueError("Boom")
            
        with pytest.raises(ValueError, match="Boom"):
            error_node({})

