# Agents Test Suite Layout

This directory now mirrors the structure of `src/agents/` so that every agent
has its own pytest package.  Splitting the mega test modules keeps intent
focused, makes fixtures re-usable, and avoids enormous files that are hard to
review.

```
tests/agents/
├── analysis/                # `results_analyzer_node`, `comparison_validator_node`
├── base/                    # shared decorators + helper utilities
├── code/                    # `code_generator_node`, `code_reviewer_node`
├── design/
├── execution/
├── planning/
├── reporting/
├── stage_selection/
├── supervision/
├── trigger_handlers/
├── user_interaction/
├── helpers/                 # low-level helper units (unchanged)
├── shared_objects.py        # constants/helpers used across suites
└── conftest.py              # shared fixtures & patch helpers
```

## Shared Fixtures & Helpers

- `tests/agents/conftest.py` exposes canonical state factories (e.g.
  `analysis_state`, `code_state`, `plan_state`, …) derived from `minimal_state`.
  All suites import these instead of redefining bespoke dictionaries.
- Common mock helpers such as `patch_agent_stack` and
  `patched_call_agent_with_metrics` live next to the fixtures so we only write
  the `build_agent_prompt/check_context_or_escalate/call_agent_with_metrics`
  patch boilerplate once.
- Reusable constants/objects (`NonSerializable`, long fallback payloads, sample
  CLI questions) reside in `tests/agents/shared_objects.py`.  Both large suites
  and the helper tests can import them without creating circular dependencies.

## Working With Slow Tests

- CLI-heavy suites (notably `user_interaction` and selected `trigger_handlers`
  flows) set `pytestmark = pytest.mark.slow`.  Register the marker in
  `pytest.ini` and run `pytest -m "not slow"` to skip them locally, or run
  `pytest tests/agents/user_interaction -m slow` to focus on them.

## Migration Checklist For New Tests

1. Pick the subpackage that matches the agent being exercised; create it if it
   does not exist yet and add an `__init__.py` to keep imports easy.
2. Import shared fixtures from `tests.agents.conftest` instead of redefining
   state dictionaries.
3. Use the patch helpers from the same module whenever you need to stub
   `call_agent_with_metrics` or the standard prompt/context functions.
4. Tag any test that performs blocking I/O or relies on signal/CLI flows with
   `@pytest.mark.slow`.

