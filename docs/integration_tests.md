# Integration Test Layout

This document explains how the integration test suite is being decomposed so
individual modules stay focused on a single workflow domain. Keep it nearby when
adding new suites or moving tests between files.

## Current Test Structure

The integration tests are organized by domain into subdirectories:

- **Planning**: `tests/integration/planning/` - Adapt prompts node, planner LLM call contracts, plan reviewer decisions, progress initialization, dependency validation, escalation paths
- **Supervision**: `tests/integration/supervision/` - Supervisor node triggers and handlers
- **Routing**: `tests/integration/routing/` - Routing functions for every review/verdict
- **Stage Selection**: `tests/integration/stage_selection/` - Stage selection edge cases
- **Execution**: `tests/integration/execution/` - Execution validators, physics sanity checks
- **Analysis**: `tests/integration/analysis/` - Results analyzer, comparison validator
- **Reporting**: `tests/integration/reporting/` - Backtrack node behaviors, report generator completeness, quantitative summaries, metrics aggregation

Smaller files that should also benefit from shared helpers: `tests/integration/test_code_agents.py`,
`tests/integration/test_design_agents.py`,
`tests/integration/test_invariants.py`, and
`tests/integration/test_paper_loader_integration.py`.

## Target Structure

```
tests/integration/
├── helpers/
│   ├── state_factories.py
│   └── agent_responses.py
├── planning/
│   ├── test_prompt_adaptation.py
│   ├── test_plan_node_contracts.py
│   ├── test_plan_reviewer_rules.py
│   └── test_planning_edge_cases.py
├── supervision/
│   └── test_supervisor_triggers.py
├── routing/
│   └── test_routing_decisions.py
├── stage_selection/
│   └── test_select_stage_node.py
├── execution/
│   └── test_execution_validator.py
├── analysis/
│   ├── test_results_analyzer.py
│   └── test_comparison_validator.py
└── reporting/
    ├── test_handle_backtrack.py
    └── test_generate_report.py
```

### Domain Split Notes

- **Planning package**: Keep adapt-prompts, plan node contracts, reviewer rules,
  edge cases, and progress initialization separate so regressions are easy to
  localize.
- **Supervision & routing**: Supervisor trigger handlers and routing limits
  share fixtures; splitting them into their own directories prevents a single
  file from mixing supervision, routing, and stage-selection behavior.
- **Execution & analysis**: Execution validators (including physics sanity) and
  analysis/comparison validators have distinct IO patterns. Separate modules
  avoid cross-contamination of mocks.
- **Reporting**: Backtracking logic and report generation are heavy suites and
  should live under `tests/integration/reporting/`.

## Shared Helpers

Create helper modules under `tests/integration/helpers/`:

- `state_factories.py`: `make_plan`, `make_progress`, and
  `augment_state(base_state, **overrides)` to consolidate the repeated dict
  construction spread across existing modules.
- `agent_responses.py`: canonical mock payload builders for planner, supervisor,
  code/design/execution reviewers, results analyzer, and report generator.

All integration suites (large and small) should import from these helpers rather
than redefining ad-hoc mocks. This keeps fixtures aligned across the split
modules and helps smaller files benefit from the shared code immediately.

## Long-Running Suites

Mark any suite that touches the filesystem or spawns real subprocesses with
`@pytest.mark.slow`. `pytest.ini` already defines the marker; we mainly need to
tag:

- `TestPaperLoaderIntegration` (real file copies)
- Results analyzer suites that write temporary CSVs
- Any future tests that run real code generation or code-runner flows

## Checklist When Moving Tests

1. Move class or function into the new module location above.
2. Replace inline fixtures/mocks with helper imports from
   `tests.integration.helpers`.
3. Update `__init__.py` files or `pytest` path configuration if a new package
   requires explicit discovery.
4. Delete the emptied sections from the original oversized files—leave behind a
   short comment pointing at the new module if necessary.
5. Run `pytest tests/integration` (or the relevant subset) to ensure the split
   did not break imports or fixtures.

Following this structure keeps each domain-specific suite under 300 LOC and
reduces duplication as the workflow evolves.

