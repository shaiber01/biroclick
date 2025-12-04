"""Shared helpers and fixtures for LLM contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

import pytest


TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_DIR.parent

SCHEMAS_DIR = REPO_ROOT / "schemas"
MOCK_RESPONSES_DIR = TESTS_DIR / "fixtures" / "mock_responses"


def load_schema(schema_name: str) -> Dict:
    """Load a JSON schema file by name."""
    path = SCHEMAS_DIR / schema_name
    with open(path, "r") as file:
        return json.load(file)


def load_mock_response(agent_name: str) -> Dict:
    """Load a mock response for the given agent."""
    path = MOCK_RESPONSES_DIR / f"{agent_name}_response.json"
    with open(path, "r") as file:
        return json.load(file)


# Schema mapping: agent name -> schema file
AGENT_SCHEMAS = {
    "planner": "planner_output_schema.json",
    "plan_reviewer": "plan_reviewer_output_schema.json",
    "simulation_designer": "simulation_designer_output_schema.json",
    "design_reviewer": "design_reviewer_output_schema.json",
    "code_generator": "code_generator_output_schema.json",
    "code_reviewer": "code_reviewer_output_schema.json",
    "execution_validator": "execution_validator_output_schema.json",
    "physics_sanity": "physics_sanity_output_schema.json",
    "results_analyzer": "results_analyzer_output_schema.json",
    "comparison_validator": "comparison_validator_output_schema.json",
    "supervisor": "supervisor_output_schema.json",
}


@pytest.fixture(scope="session")
def schema_loader() -> Callable[[str], Dict]:
    """Pytest fixture exposing the schema loader helper."""
    return load_schema


@pytest.fixture(scope="session")
def mock_response_loader() -> Callable[[str], Dict]:
    """Pytest fixture exposing the mock response loader helper."""
    return load_mock_response



