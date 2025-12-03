"""
Shared helpers for trigger handler tests.
"""


def result_has_value(result, key, value):
    """Helper to check a value in the result dict."""
    return result.get(key) == value

