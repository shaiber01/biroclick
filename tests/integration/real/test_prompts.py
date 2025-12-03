"""Integration tests ensuring every agent prompt exists and loads."""

import pytest

from schemas.state import create_initial_state
from src.prompts import PROMPTS_DIR, build_agent_prompt


class TestPromptFilesExist:
    """Verify all prompt files referenced in code actually exist."""

    AGENTS_REQUIRING_PROMPTS = [
        "prompt_adaptor",
        "planner",
        "plan_reviewer",
        "simulation_designer",
        "design_reviewer",
        "code_generator",
        "code_reviewer",
        "execution_validator",
        "physics_sanity",
        "results_analyzer",
        "supervisor",
        "report_generator",
    ]

    @pytest.mark.parametrize("agent_name", AGENTS_REQUIRING_PROMPTS)
    def test_prompt_file_exists(self, agent_name):
        """Each agent's prompt file must exist on disk."""
        prompt_file = PROMPTS_DIR / f"{agent_name}_agent.md"
        assert prompt_file.exists(), (
            f"Missing prompt file: {prompt_file}\n"
            f"Agent '{agent_name}' calls build_agent_prompt() but the file doesn't exist.\n"
            "This will crash at runtime!"
        )

    @pytest.mark.parametrize("agent_name", AGENTS_REQUIRING_PROMPTS)
    def test_prompt_file_not_empty(self, agent_name):
        """Each prompt file must have content."""
        prompt_file = PROMPTS_DIR / f"{agent_name}_agent.md"
        if prompt_file.exists():
            content = prompt_file.read_text()
            assert len(content) > 100, (
                f"Prompt file {prompt_file} is suspiciously short ({len(content)} chars).\n"
                "Expected substantial prompt content."
            )

    @pytest.mark.parametrize("agent_name", AGENTS_REQUIRING_PROMPTS)
    def test_build_agent_prompt_succeeds(self, agent_name):
        """build_agent_prompt() must succeed for each agent."""
        state = create_initial_state(
            paper_id="test",
            paper_text="Test paper content for validation.",
            paper_domain="plasmonics",
        )

        try:
            prompt = build_agent_prompt(agent_name, state)
            assert prompt is not None
            assert len(prompt) > 0
        except FileNotFoundError as exc:
            pytest.fail(f"build_agent_prompt('{agent_name}') failed: {exc}")


