"""Shared fixtures for the split code runner test suite."""

import shutil
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def code_runner_state_factory():
    """
    Provides a factory that builds a baseline state dict for `run_code_node`.
    Automatically tracks `paper_id` paths under `outputs/` and cleans them up
    after the test finishes to avoid test pollution.
    """

    paper_ids = []

    def _factory(**overrides):
        stage_id = overrides.get("current_stage_id", "stage1")
        runtime_budget = overrides.pop("runtime_budget_minutes", 10)
        plan = overrides.pop(
            "plan",
            {"stages": [{"stage_id": stage_id, "runtime_budget_minutes": runtime_budget}]},
        )
        state = {
            "code": overrides.pop("code", "import meep"),
            "current_stage_id": stage_id,
            "paper_id": overrides.pop("paper_id", f"test_paper_{uuid4().hex}"),
            "plan": plan,
            "runtime_config": overrides.pop(
                "runtime_config",
                {"max_memory_gb": 16.0, "max_cpu_cores": 8},
            ),
        }

        state.update(overrides)
        paper_id = state.get("paper_id")
        if paper_id:
            paper_ids.append(paper_id)

        return state

    yield _factory

    for paper_id in paper_ids:
        output_dir = Path("outputs") / paper_id
        if output_dir.exists():
            shutil.rmtree(output_dir)

