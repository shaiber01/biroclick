# Agents Refactoring - Remaining Next Steps

This file tracks remaining improvements after the initial agents.py refactoring.

## Coverage Report Summary (December 2024)

Current coverage: **69%** for `src/agents/`

| Module | Coverage | Notes |
|--------|----------|-------|
| `__init__.py` | 100% | ✅ |
| `constants.py` | 100% | ✅ |
| `helpers/stubs.py` | 100% | ✅ |
| `helpers/__init__.py` | 100% | ✅ |
| `helpers/validation.py` | 95% | ✅ |
| `design.py` | 95% | ✅ |
| `helpers/metrics.py` | 92% | ✅ |
| `helpers/materials.py` | 91% | ✅ |
| `planning.py` | 91% | ✅ |
| `helpers/context.py` | 89% | ✅ |
| `helpers/numeric.py` | 86% | ✅ |
| `reporting.py` | 83% | ⚠️ |
| `code.py` | 82% | ⚠️ |
| `execution.py` | 82% | ⚠️ |
| `user_interaction.py` | 59% | ⚠️ Interactive CLI paths |
| `stage_selection.py` | 56% | ⚠️ Complex dependency logic |
| `analysis.py` | 32% | ❌ Needs more tests |
| `supervision.py` | 31% | ❌ Many trigger handlers |
| `base.py` | 0% | ❌ Decorator unused |

### Priority: Improve Test Coverage

The following modules need additional test coverage:

1. **`supervision.py`** (31%) - Many trigger handler functions untested
2. **`analysis.py`** (32%) - Results analysis and comparison validation
3. **`stage_selection.py`** (56%) - Complex dependency and deadlock detection
4. **`user_interaction.py`** (59%) - CLI-based paths hard to test

To regenerate the HTML coverage report:
```bash
pytest --cov=src/agents --cov-report=html tests/
# Then open htmlcov/index.html
```

## Low Priority (Future Improvements)

### 1. Extract Common Patterns Further
Some patterns repeat across nodes:
- LLM error handling (try/except with auto-approve fallback)
- Revision counter incrementing
- Context check handling

These could potentially use the `with_context_check` decorator pattern more consistently.

### 2. Split Large Modules
`supervision.py` and `analysis.py` are still relatively large. If they continue to grow, consider further decomposition.

### 3. Add Integration Tests for Full Workflows
The current integration tests mock LLM calls. Consider adding end-to-end tests with recorded LLM responses for regression testing.

### 4. Use `base.py` Decorator
The `with_context_check` decorator in `base.py` is currently unused. Consider refactoring nodes to use it for consistent context handling.

---

*Created after agents.py refactoring - December 2024*
