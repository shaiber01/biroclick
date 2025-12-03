"""Fixtures shared across LLM client tests."""

import pytest

from src.llm_client import reset_llm_client


@pytest.fixture
def fresh_llm_client():
    """Ensure a clean LLM client instance before and after each test."""
    reset_llm_client()
    yield
    reset_llm_client()



