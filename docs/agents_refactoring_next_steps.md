# Agents Refactoring - Remaining Next Steps

This file tracks remaining improvements after the initial agents.py refactoring.

## ✅ Coverage Report Summary (December 2024)

**Current coverage: 88%** for `src/agents/` (improved from 69%)

| Module | Coverage | Notes |
|--------|----------|-------|
| `__init__.py` | 100% | ✅ |
| `constants.py` | 100% | ✅ |
| `helpers/stubs.py` | 100% | ✅ |
| `helpers/__init__.py` | 100% | ✅ |
| `helpers/validation.py` | 96% | ✅ |
| `design.py` | 95% | ✅ |
| `helpers/metrics.py` | 92% | ✅ |
| `helpers/materials.py` | 91% | ✅ |
| `planning.py` | 91% | ✅ |
| `stage_selection.py` | 89% | ✅ (was 56%) |
| `supervision.py` | 89% | ✅ (was 31%) |
| `helpers/context.py` | 89% | ✅ |
| `user_interaction.py` | 87% | ✅ (was 59%) |
| `helpers/numeric.py` | 86% | ✅ |
| `reporting.py` | 83% | ✅ |
| `analysis.py` | 82% | ✅ (was 32%) |
| `code.py` | 82% | ⚠️ |
| `execution.py` | 82% | ⚠️ |
| `base.py` | 0% | ❌ Decorator unused |

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

## Low Priority (Future Improvements)

### 2. Extract Common Patterns Further
Some patterns repeat across nodes:
- LLM error handling (try/except with auto-approve fallback)
- Revision counter incrementing

### 3. Split Large Modules
`supervision.py` and `analysis.py` are still relatively large. If they continue to grow, consider further decomposition.

### 4. Add Integration Tests for Full Workflows
The current integration tests mock LLM calls. Consider adding end-to-end tests with recorded LLM responses for regression testing.

---

*Created after agents.py refactoring - December 2024*
*Updated with improved coverage - December 2024*
