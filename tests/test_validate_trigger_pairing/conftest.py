"""
Fixtures for validate_trigger_pairing tool tests.

These fixtures provide synthetic Python code samples to test the AST analyzer
and pairing rule validation.
"""

import pytest


# =============================================================================
# Valid Code Samples (should pass)
# =============================================================================

@pytest.fixture
def valid_dict_literal():
    """Code with properly paired dict literal."""
    return '''
def handle_limit():
    return {
        "ask_user_trigger": "test_trigger",
        "pending_user_questions": ["What would you like to do?"],
        "last_node_before_ask_user": "test_node",
    }
'''


@pytest.fixture
def valid_dict_literal_multikey():
    """Code with multiple keys including both trigger and questions."""
    return '''
def handle_error(state):
    result = {
        "workflow_phase": "error_handling",
        "ask_user_trigger": "execution_failure_limit",
        "pending_user_questions": [
            "Execution failed. How should we proceed?",
            "Options: RETRY, SKIP, ABORT",
        ],
        "last_node_before_ask_user": "execution_check",
    }
    return result
'''


@pytest.fixture
def valid_subscript():
    """Code with properly paired subscript assignments."""
    return '''
def handle_failure(state):
    result = {}
    result["ask_user_trigger"] = "physics_failure_limit"
    result["pending_user_questions"] = [
        "Physics check failed. Please review."
    ]
    result["last_node_before_ask_user"] = "physics_check"
    return result
'''


@pytest.fixture
def valid_subscript_nearby():
    """Code with subscript assignments within 10 lines of each other."""
    return '''
def handle_complex_failure(state, error):
    result = {}
    
    # Set up the trigger
    result["ask_user_trigger"] = "complex_error"
    
    # Log some information
    logger.info(f"Error occurred: {error}")
    
    # Now set the questions
    result["pending_user_questions"] = ["An error occurred. How to proceed?"]
    
    return result
'''


@pytest.fixture
def clearing_trigger():
    """Clearing trigger with None - should pass without needing questions."""
    return '''
def clear_user_interaction(result):
    result["ask_user_trigger"] = None
    result["pending_user_questions"] = []
    result["user_responses"] = {}
'''


@pytest.fixture
def clearing_trigger_alone():
    """Clearing trigger with None alone - OK because it's clearing."""
    return '''
def supervisor_continue(result):
    result["supervisor_verdict"] = "ok_continue"
    result["ask_user_trigger"] = None
'''


@pytest.fixture
def preserving_trigger():
    """Preserving existing trigger from state - should pass."""
    return '''
def supervisor_node(state):
    ask_user_trigger = state.get("ask_user_trigger")
    result = {}
    # ... processing ...
    result["ask_user_trigger"] = ask_user_trigger
    return result
'''


@pytest.fixture
def preserving_trigger_param():
    """Preserving trigger passed as parameter - should pass."""
    return '''
def handle_with_trigger(trigger):
    result = {}
    result["ask_user_trigger"] = trigger
    return result
'''


# =============================================================================
# Invalid Code Samples (should fail)
# =============================================================================

@pytest.fixture
def invalid_dict_literal():
    """Code with unpaired dict literal - should fail."""
    return '''
def broken_handler():
    return {
        "ask_user_trigger": "test_trigger",
        # Missing pending_user_questions!
        "last_node_before_ask_user": "test_node",
    }
'''


@pytest.fixture
def invalid_dict_literal_empty_questions():
    """Dict literal with trigger but no questions key at all."""
    return '''
def another_broken_handler():
    return {
        "workflow_phase": "error",
        "ask_user_trigger": "some_error",
    }
'''


@pytest.fixture
def invalid_subscript_no_questions():
    """Subscript assignment without nearby questions - should fail."""
    return '''
def broken_subscript_handler(state):
    result = {}
    result["ask_user_trigger"] = "orphan_trigger"
    # No pending_user_questions anywhere nearby!
    result["workflow_phase"] = "stuck"
    return result
'''


@pytest.fixture
def invalid_subscript_far_questions():
    """Subscript with questions more than 10 lines away - should fail."""
    return '''
def broken_far_questions(state):
    result = {}
    result["ask_user_trigger"] = "distant_trigger"
    
    # Lots of unrelated code
    x = 1
    y = 2
    z = 3
    a = 4
    b = 5
    c = 6
    d = 7
    e = 8
    f = 9
    g = 10
    h = 11
    
    # Questions are too far away (>10 lines)
    result["pending_user_questions"] = ["Finally!"]
    return result
'''


# =============================================================================
# Edge Cases
# =============================================================================

@pytest.fixture
def suspicious_variable():
    """Variable assignment that's neither preservation nor obvious - warning."""
    return '''
def handle_dynamic(state, trigger_name):
    result = {}
    result["ask_user_trigger"] = trigger_name  # Not a known preservation var
    return result
'''


@pytest.fixture
def multiple_triggers_same_function():
    """Multiple trigger assignments in same function."""
    return '''
def multi_trigger_handler(state, condition):
    result = {}
    
    if condition == "error_a":
        result["ask_user_trigger"] = "error_a_trigger"
        result["pending_user_questions"] = ["Error A occurred"]
    elif condition == "error_b":
        result["ask_user_trigger"] = "error_b_trigger"
        result["pending_user_questions"] = ["Error B occurred"]
    else:
        result["ask_user_trigger"] = None
    
    return result
'''


@pytest.fixture
def nested_dict_literal():
    """Trigger in nested dict - should be detected."""
    return '''
def nested_handler():
    outer = {
        "inner": {
            "ask_user_trigger": "nested_trigger",
            "pending_user_questions": ["Nested question"],
        }
    }
    return outer
'''


@pytest.fixture
def invalid_nested_dict():
    """Invalid trigger in nested dict without questions."""
    return '''
def invalid_nested():
    outer = {
        "inner": {
            "ask_user_trigger": "orphan_nested",
        }
    }
    return outer
'''


@pytest.fixture
def class_method_trigger():
    """Trigger assignment inside a class method."""
    return '''
class Handler:
    def handle_error(self, state):
        result = {
            "ask_user_trigger": "class_trigger",
            "pending_user_questions": ["Class method question"],
        }
        return result
'''


@pytest.fixture
def async_function_trigger():
    """Trigger assignment in async function."""
    return '''
async def async_handler(state):
    result = {
        "ask_user_trigger": "async_trigger",
        "pending_user_questions": ["Async question"],
    }
    return result
'''


@pytest.fixture
def syntax_error_code():
    """Code with syntax error - should be handled gracefully."""
    return '''
def broken_syntax(
    # Missing closing paren
    result = {}
'''


@pytest.fixture
def empty_file():
    """Empty file - should be handled gracefully."""
    return ''


@pytest.fixture
def no_triggers():
    """File with no triggers at all."""
    return '''
def normal_function(x, y):
    return x + y

def another_function():
    result = {"status": "ok"}
    return result
'''
