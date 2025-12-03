"""Tests for with_context_check decorator."""

import pytest
from unittest.mock import patch, call

from src.agents.base import with_context_check

class TestWithContextCheck:
    """Tests for with_context_check decorator."""

    @patch("src.agents.base.check_context_or_escalate")
    def test_returns_escalation_when_awaiting_input(self, mock_check):
        """Should return escalation when context check requires user input."""
        escalation = {
            "awaiting_user_input": True,
            "pending_user_questions": ["Context overflow"],
            "ask_user_trigger": "context_overflow",
            "last_node_before_ask_user": "test_node",
        }
        mock_check.return_value = escalation
        
        original_state = {"some": "state", "data": "preserved"}
        func_called = []
        
        @with_context_check("test_node")
        def test_node(state):
            func_called.append(state)
            return {"result": "success"}
        
        result = test_node(original_state)
        
        # Verify escalation is returned exactly as-is
        assert result == escalation
        assert result["awaiting_user_input"] is True
        assert result["pending_user_questions"] == ["Context overflow"]
        assert result["ask_user_trigger"] == "context_overflow"
        assert result["last_node_before_ask_user"] == "test_node"
        
        # Verify function was NOT called when awaiting input
        assert len(func_called) == 0
        
        # Verify check was called with correct arguments
        mock_check.assert_called_once_with(original_state, "test_node")

    @patch("src.agents.base.check_context_or_escalate")
    def test_returns_escalation_when_awaiting_input_false_but_present(self, mock_check):
        """Should merge state updates when awaiting_user_input is False."""
        mock_check.return_value = {
            "awaiting_user_input": False,
            "metrics": {"tokens": 100},
        }
        
        original_state = {"original": "data", "preserved": True}
        received_state = []
        
        @with_context_check("test_node")
        def test_node(state):
            received_state.append(state.copy())
            return {"result": "success", "metrics": state.get("metrics")}
        
        result = test_node(original_state)
        
        # Function should be called with merged state
        assert len(received_state) == 1
        assert received_state[0]["original"] == "data"
        assert received_state[0]["preserved"] is True
        assert received_state[0]["metrics"] == {"tokens": 100}
        assert received_state[0]["awaiting_user_input"] is False
        
        # Result should come from function, not escalation
        assert result["result"] == "success"
        assert result["metrics"] == {"tokens": 100}

    @patch("src.agents.base.check_context_or_escalate")
    def test_updates_state_before_calling_func(self, mock_check):
        """Should update state with context changes BEFORE calling function."""
        state_updates = {"metrics": {"tokens": 100}, "context_budget": 5000}
        mock_check.return_value = state_updates
        
        original_state = {"original": "data", "preserved": "value"}
        received_state = []
        
        @with_context_check("test_node")
        def test_node(state):
            received_state.append(state.copy())
            # Verify state has been updated inside function
            assert state.get("metrics") == {"tokens": 100}
            assert state.get("context_budget") == 5000
            assert state.get("original") == "data"
            assert state.get("preserved") == "value"
            return {"result": "success"}
        
        result = test_node(original_state)
        
        # Verify function received merged state
        assert len(received_state) == 1
        assert received_state[0]["metrics"] == {"tokens": 100}
        assert received_state[0]["context_budget"] == 5000
        assert received_state[0]["original"] == "data"
        assert received_state[0]["preserved"] == "value"
        
        # Verify result from function
        assert result["result"] == "success"
        
        # Verify original state was not mutated
        assert original_state == {"original": "data", "preserved": "value"}

    @patch("src.agents.base.check_context_or_escalate")
    def test_state_updates_override_existing_keys(self, mock_check):
        """Should override existing state keys when merging updates."""
        mock_check.return_value = {"metrics": {"tokens": 200}, "data": "updated"}
        
        original_state = {"metrics": {"tokens": 50}, "data": "original"}
        received_state = []
        
        @with_context_check("test_node")
        def test_node(state):
            received_state.append(state.copy())
            return {"metrics": state.get("metrics")}
        
        result = test_node(original_state)
        
        # Verify metrics were overridden
        assert received_state[0]["metrics"] == {"tokens": 200}
        assert received_state[0]["data"] == "updated"
        assert result["metrics"] == {"tokens": 200}

    @patch("src.agents.base.check_context_or_escalate")
    def test_continues_when_no_escalation(self, mock_check):
        """Should continue to wrapped function when no escalation needed."""
        mock_check.return_value = None
        
        original_state = {"data": "test_value", "workflow_phase": "initial"}
        received_state = []
        
        @with_context_check("test_node")
        def test_node(state):
            received_state.append(state.copy())
            return {"workflow_phase": "testing", "data": state.get("data")}
        
        result = test_node(original_state)
        
        # Verify function received original state unchanged
        assert len(received_state) == 1
        assert received_state[0] == original_state
        
        # Verify result from function
        assert result["workflow_phase"] == "testing"
        assert result["data"] == "test_value"

    @patch("src.agents.base.check_context_or_escalate")
    def test_continues_when_empty_dict_escalation(self, mock_check):
        """Should continue when escalation is empty dict (not None)."""
        mock_check.return_value = {}
        
        original_state = {"data": "test"}
        received_state = []
        
        @with_context_check("test_node")
        def test_node(state):
            received_state.append(state.copy())
            return {"result": "success"}
        
        result = test_node(original_state)
        
        # Empty dict should not have awaiting_user_input, so function should be called
        assert len(received_state) == 1
        assert received_state[0] == original_state
        assert result["result"] == "success"

    @patch("src.agents.base.check_context_or_escalate")
    def test_calls_check_with_correct_node_name(self, mock_check):
        """Should call check_context_or_escalate with correct node name."""
        mock_check.return_value = None
        
        test_state = {"state": "data"}
        
        @with_context_check("my_custom_node")
        def my_node(state):
            return {}
        
        my_node(test_state)
        
        # Verify check was called exactly once with correct arguments
        mock_check.assert_called_once()
        call_args = mock_check.call_args
        assert call_args[0][0] == test_state  # First arg is state
        assert call_args[0][1] == "my_custom_node"  # Second arg is node_name
        assert len(call_args[0]) == 2  # Only positional args, no kwargs

    @patch("src.agents.base.check_context_or_escalate")
    def test_calls_check_with_correct_state(self, mock_check):
        """Should pass the exact state object to check_context_or_escalate."""
        mock_check.return_value = None
        
        test_state = {"key": "value", "nested": {"data": 123}}
        
        @with_context_check("test_node")
        def test_node(state):
            return {}
        
        test_node(test_state)
        
        # Verify state was passed correctly
        mock_check.assert_called_once()
        passed_state = mock_check.call_args[0][0]
        assert passed_state == test_state
        assert passed_state is test_state  # Should be same object reference

    @patch("src.agents.base.check_context_or_escalate")
    def test_preserves_function_metadata(self, mock_check):
        """Should preserve function name and docstring."""
        mock_check.return_value = None
        
        @with_context_check("test")
        def my_documented_function(state):
            """This is my docstring."""
            return {}
        
        assert my_documented_function.__name__ == "my_documented_function"
        assert my_documented_function.__doc__ == "This is my docstring."
        assert "docstring" in my_documented_function.__doc__

    @patch("src.agents.base.check_context_or_escalate")
    def test_supports_kwargs_passing(self, mock_check):
        """Should support passing additional keyword arguments to the decorated function."""
        mock_check.return_value = None
        
        received_kwargs = []
        
        @with_context_check("test_kwargs")
        def node_with_args(state, extra_arg=None, another_arg=None):
            received_kwargs.append({"extra_arg": extra_arg, "another_arg": another_arg})
            return {"extra": extra_arg, "another": another_arg}
        
        result = node_with_args({"state": "data"}, extra_arg="working", another_arg="also_working")
        
        assert len(received_kwargs) == 1
        assert received_kwargs[0]["extra_arg"] == "working"
        assert received_kwargs[0]["another_arg"] == "also_working"
        assert result["extra"] == "working"
        assert result["another"] == "also_working"

    @patch("src.agents.base.check_context_or_escalate")
    def test_supports_args_passing(self, mock_check):
        """Should support passing additional positional arguments to the decorated function."""
        mock_check.return_value = None
        
        received_args = []
        
        @with_context_check("test_args")
        def node_with_args(state, *args):
            received_args.append(list(args))
            return {"args": list(args)}
        
        result = node_with_args({"state": "data"}, "arg1", "arg2", "arg3")
        
        assert len(received_args) == 1
        assert received_args[0] == ["arg1", "arg2", "arg3"]
        assert result["args"] == ["arg1", "arg2", "arg3"]

    @patch("src.agents.base.check_context_or_escalate")
    def test_supports_args_and_kwargs_together(self, mock_check):
        """Should support both args and kwargs together."""
        mock_check.return_value = None
        
        received_args_kwargs = []
        
        @with_context_check("test_both")
        def node_with_both(state, *args, **kwargs):
            received_args_kwargs.append({"args": list(args), "kwargs": kwargs})
            return {"args": list(args), "kwargs": kwargs}
        
        result = node_with_both({"state": "data"}, "pos1", "pos2", kw1="val1", kw2="val2")
        
        assert len(received_args_kwargs) == 1
        assert received_args_kwargs[0]["args"] == ["pos1", "pos2"]
        assert received_args_kwargs[0]["kwargs"] == {"kw1": "val1", "kw2": "val2"}
        assert result["args"] == ["pos1", "pos2"]
        assert result["kwargs"] == {"kw1": "val1", "kw2": "val2"}

    @patch("src.agents.base.check_context_or_escalate")
    def test_propagates_exceptions(self, mock_check):
        """Should propagate exceptions from the decorated function."""
        mock_check.return_value = None
        
        @with_context_check("test_error")
        def error_node(state):
            raise ValueError("Boom")
        
        with pytest.raises(ValueError, match="Boom"):
            error_node({})
        
        # Verify check was still called before exception
        mock_check.assert_called_once()

    @patch("src.agents.base.check_context_or_escalate")
    def test_propagates_exceptions_from_check(self, mock_check):
        """Should propagate exceptions from check_context_or_escalate."""
        mock_check.side_effect = RuntimeError("Check failed")
        
        @with_context_check("test_error")
        def test_node(state):
            return {"result": "success"}
        
        with pytest.raises(RuntimeError, match="Check failed"):
            test_node({})
        
        # Function should not be called if check raises
        mock_check.assert_called_once()

    @patch("src.agents.base.check_context_or_escalate")
    def test_handles_empty_state(self, mock_check):
        """Should handle empty state dictionary."""
        mock_check.return_value = None
        
        @with_context_check("test_empty")
        def test_node(state):
            assert isinstance(state, dict)
            assert len(state) == 0
            return {"result": "success"}
        
        result = test_node({})
        assert result["result"] == "success"
        mock_check.assert_called_once_with({}, "test_empty")

    @patch("src.agents.base.check_context_or_escalate")
    def test_handles_nested_state_updates(self, mock_check):
        """Should handle nested dictionary updates correctly."""
        mock_check.return_value = {
            "metrics": {"tokens": 100, "cost": 0.05},
            "nested": {"level1": {"level2": "value"}},
        }
        
        original_state = {
            "metrics": {"tokens": 50},
            "other": "data",
        }
        received_state = []
        
        @with_context_check("test_nested")
        def test_node(state):
            received_state.append(state.copy())
            return {"metrics": state.get("metrics")}
        
        result = test_node(original_state)
        
        # Verify nested updates replace entire nested dicts (not merge)
        assert received_state[0]["metrics"] == {"tokens": 100, "cost": 0.05}
        assert received_state[0]["nested"] == {"level1": {"level2": "value"}}
        assert received_state[0]["other"] == "data"
        assert result["metrics"] == {"tokens": 100, "cost": 0.05}

    @patch("src.agents.base.check_context_or_escalate")
    def test_awaiting_input_with_missing_key(self, mock_check):
        """Should handle escalation dict without awaiting_user_input key."""
        # If key is missing, get() returns None, so should merge and call function
        mock_check.return_value = {
            "pending_user_questions": ["Question"],
            # No awaiting_user_input key
        }
        
        received_state = []
        
        @with_context_check("test_missing_key")
        def test_node(state):
            received_state.append(state.copy())
            return {"result": "success"}
        
        result = test_node({"data": "test"})
        
        # Should call function since awaiting_user_input is None (missing)
        assert len(received_state) == 1
        assert received_state[0]["pending_user_questions"] == ["Question"]
        assert result["result"] == "success"

    @patch("src.agents.base.check_context_or_escalate")
    def test_multiple_decorated_functions(self, mock_check):
        """Should work correctly with multiple decorated functions."""
        mock_check.return_value = None
        
        @with_context_check("node1")
        def node1(state):
            return {"node": 1}
        
        @with_context_check("node2")
        def node2(state):
            return {"node": 2}
        
        result1 = node1({"data": "test1"})
        result2 = node2({"data": "test2"})
        
        assert result1["node"] == 1
        assert result2["node"] == 2
        assert mock_check.call_count == 2
        assert mock_check.call_args_list[0][0][1] == "node1"
        assert mock_check.call_args_list[1][0][1] == "node2"

    @patch("src.agents.base.check_context_or_escalate")
    def test_function_return_value_preserved(self, mock_check):
        """Should preserve exact return value from wrapped function."""
        mock_check.return_value = None
        
        complex_return = {
            "workflow_phase": "testing",
            "data": {"nested": "value"},
            "list": [1, 2, 3],
            "none_value": None,
        }
        
        @with_context_check("test_return")
        def test_node(state):
            return complex_return
        
        result = test_node({"state": "data"})
        
        # Verify return value is preserved exactly
        assert result == complex_return
        assert result["data"]["nested"] == "value"
        assert result["list"] == [1, 2, 3]
        assert result["none_value"] is None
        # Verify it's the same object reference (normal Python behavior)
        assert result is complex_return

