# Agents Refactoring - Remaining Next Steps

This file tracks remaining improvements after the initial agents.py refactoring.

## ✅ Coverage Report Summary (December 2024)

**Current coverage: 89%** for `src/agents/` (improved from 69%)

| Module | Coverage | Notes |
|--------|----------|-------|
| `__init__.py` | 100% | ✅ |
| `base.py` | 100% | ✅ |
| `constants.py` | 100% | ✅ |
| `helpers/stubs.py` | 100% | ✅ |
| `helpers/__init__.py` | 100% | ✅ |
| `helpers/validation.py` | 96% | ✅ |
| `design.py` | 94% | ✅ |
| `helpers/metrics.py` | 92% | ✅ |
| `helpers/materials.py` | 91% | ✅ |
| `planning.py` | 91% | ✅ |
| `stage_selection.py` | 89% | ✅ (was 56%) |
| `supervision/supervisor.py` | 89% | ✅ (split from monolith) |
| `supervision/trigger_handlers.py` | 84% | ✅ (split from monolith) |
| `helpers/context.py` | 89% | ✅ |
| `user_interaction.py` | 87% | ✅ (was 59%) |
| `helpers/numeric.py` | 86% | ✅ |
| `execution.py` | 84% | ✅ |
| `reporting.py` | 83% | ✅ |
| `analysis.py` | 82% | ✅ (was 32%) |
| `code.py` | 80% | ⚠️ |

To regenerate the HTML coverage report:
```bash
pytest --cov=src/agents --cov-report=html tests/
# Then open htmlcov/index.html
```

## ✅ Completed Items

### Use `base.py` Decorator (Completed)
The `with_context_check` decorator from `base.py` is now used by the following nodes for consistent context handling:
- `code_reviewer_node`
- `execution_validator_node`
- `physics_sanity_node`
- `design_reviewer_node`
- `comparison_validator_node`
- `adapt_prompts_node`
- `plan_reviewer_node`

`base.py` now has 100% test coverage (was 0%).

### Extract Common Patterns (Completed)
Extracted common patterns into `base.py`:

1. **`increment_counter_with_max(state, counter_name, config_key, default_max)`**
   - Handles bounded counter incrementing with configurable max
   - Returns `(new_count, was_incremented)` tuple
   - Used by: `code_reviewer`, `design_reviewer`, `execution_validator`, `physics_sanity`, `comparison_validator`

2. **`create_llm_error_auto_approve(agent_name, error, default_verdict, error_truncate_len)`**
   - Creates auto-approve response for reviewer LLM failures
   - Used by: `code_reviewer`, `design_reviewer`, `plan_reviewer`, `execution_validator`, `physics_sanity`

3. **`create_llm_error_escalation(agent_name, workflow_phase, error, error_truncate_len)`**
   - Creates user escalation response for critical LLM failures
   - Used by: `code_generator`, `simulation_designer`, `plan_node`

4. **`create_llm_error_fallback(agent_name, default_verdict, feedback_msg, error_truncate_len)`**
   - Creates fallback handler for non-critical LLM failures
   - Available for: supervisor and similar nodes

17 new tests added for these utilities in `test_base.py`.

### Split Large Modules (Completed)
`supervision.py` was split into a `supervision/` package:

```
src/agents/supervision/
├── __init__.py           # Re-exports supervisor_node and handlers
├── supervisor.py         # Main supervisor_node logic (136 lines)
└── trigger_handlers.py   # Individual trigger handlers (210 lines)
```

**Benefits:**
- Each trigger handler is now a separate, testable function
- `handle_trigger()` dispatcher provides clean routing
- Shared utilities (`APPROVAL_KEYWORDS`, `_archive_with_error_handling`) reduce duplication
- `analysis.py` reviewed but not split - its linear flow and existing helper extraction make further splitting unnecessary

### Comprehensive Testing Suite (Completed)
Added three new test files with comprehensive coverage:

**1. `test_workflow_e2e.py` - Full Workflow Tests (58 tests)**
- `TestFullWorkflowSuccess`: Complete single-stage workflow
- `TestWorkflowWithRevisions`: Revision cycles with feedback
- `TestWorkflowWithFailures`: Execution failure recovery
- `TestMultiStageWorkflow`: Dependency-based stage progression
- `TestSupervisorDecisions`: Routing logic validation
- `MockResponseFactory`: Reusable factory for creating coordinated mock responses

**2. `test_llm_contracts.py` - Contract & Schema Tests (58 tests)**
- `TestMockResponsesConformToSchemas`: Type validation against JSON schemas
- `TestSchemaRequiredFields`: Required field enforcement
- `TestSchemaEnumConstraints`: Enum value validation
- `TestReviewerContract`: Verdict → routing contracts
- `TestSupervisorContract`: Decision → action contracts
- `TestPlannerContract`: Output structure for downstream nodes
- `TestAnalyzerContract`: Classification → next steps
- `TestEdgeCaseResponses`: Empty arrays, minimal responses
- `TestMalformedResponses`: Missing fields, wrong types

**3. `test_llm_smoke.py` - Real LLM Smoke Tests (12 tests, skipped without API key)**
- `TestLLMIntegration`: Module imports, prompt building
- `TestPlannerSmoke`: Real planner response validation
- `TestReviewerSmoke`: Real reviewer verdict validation
- `TestSchemaValidationSmoke`: Real outputs against schemas
- `TestLLMPerformance`: Response time checks
- `TestLLMErrorHandling`: Empty/long paper handling

Run smoke tests with: `pytest -m smoke tests/test_llm_smoke.py`

### Mock Response Cleanup (Completed)
Fixed legacy mock responses to match JSON schemas:
- `simulation_designer_response.json`: Changed `materials` from object to array format
- `results_analyzer_response.json`: Changed `confidence_factors` from object to array format
- Added `pytest.ini` with custom mark registration (`smoke`, `slow`)

## Low Priority (Future Improvements)

### Record LLM Responses for Regression
Consider recording actual LLM responses for complex scenarios to catch prompt regressions.

---

*Created after agents.py refactoring - December 2024*
*Updated with improved coverage - December 2024*
