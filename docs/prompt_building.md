# Prompt Building System

This document explains how agent prompts are constructed at runtime with context injection.

## Overview

The prompt building system (`src/prompts.py`) ensures:
1. **Single source of truth** for constants (thresholds, limits)
2. **Agent-specific context** - each agent sees only what it needs
3. **Runtime injection** - state is always current when prompts are built

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PROMPT CONSTRUCTION FLOW                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Load global_rules.md                                            │
│     └─ Contains non-negotiable rules for all agents                 │
│                                                                     │
│  2. Load agent-specific prompt (e.g., planner_agent.md)             │
│     └─ Contains role, checklists, output format                     │
│                                                                     │
│  3. Inject constants from state.py                                  │
│     └─ {THRESHOLDS_TABLE} → actual markdown table                   │
│     └─ {MAX_DESIGN_REVISIONS} → "3"                                 │
│                                                                     │
│  4. Inject state context (agent-specific)                           │
│     └─ PlannerAgent: full paper_text, paper_figures                 │
│     └─ DesignerAgent: current_stage, assumptions, feedback          │
│     └─ AnalyzerAgent: stage_outputs, target_figures                 │
│                                                                     │
│  5. Return complete prompt                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Usage

```python
from src.prompts import build_prompt

# Build a complete prompt for an agent
prompt = build_prompt(
    agent_name="planner",     # Which agent
    state=current_state,      # Current ReproState
    include_global_rules=True # Prepend global rules (default: True)
)

# Send to LLM
response = llm.invoke(prompt)
```

## How It Works

### Step 1: Load Templates

```python
from src.prompts import load_agent_prompt, load_global_rules

# Load raw templates
global_rules = load_global_rules()  # prompts/global_rules.md
agent_prompt = load_agent_prompt("planner")  # prompts/planner_agent.md
```

### Step 2: Inject Constants

Placeholders in prompts are replaced with values from `schemas/state.py`:

| Placeholder | Source | Value |
|-------------|--------|-------|
| `{THRESHOLDS_TABLE}` | `format_thresholds_table()` | Markdown table of discrepancy thresholds |
| `{MAX_DESIGN_REVISIONS}` | `MAX_DESIGN_REVISIONS` | `"3"` |
| `{MAX_ANALYSIS_REVISIONS}` | `MAX_ANALYSIS_REVISIONS` | `"2"` |
| `{MAX_REPLANS}` | `MAX_REPLANS` | `"2"` |

**Why this approach?**
- Constants defined once in Python code
- No duplication between code and prompts
- Changes propagate automatically

```python
from src.prompts import inject_constants

template = "Maximum revisions: {MAX_DESIGN_REVISIONS}"
result = inject_constants(template)
# result = "Maximum revisions: 3"
```

### Step 3: Inject State Context

Each agent receives different context based on its needs:

```python
from src.prompts import inject_state_context

prompt = inject_state_context(template, "planner", state)
# Appends paper_text, paper_figures, etc. for PlannerAgent
```

## Runtime & Hardware Configuration

Agents that need runtime or hardware information access it from state:

```python
# In src/prompts.py context builders

def _build_designer_context(state: Dict[str, Any]) -> str:
    """Build context for SimulationDesignerAgent."""
    runtime_config = state.get("runtime_config", DEFAULT_RUNTIME_CONFIG)
    hardware_config = state.get("hardware_config", DEFAULT_HARDWARE_CONFIG)
    
    debug_mode = runtime_config.get("debug_mode", False)
    
    context = f"""
### Runtime Configuration:
- **Debug Mode**: {"ENABLED ⚡" if debug_mode else "DISABLED"}
- **Resolution Factor**: {runtime_config.get('debug_resolution_factor', 1.0) if debug_mode else "1.0 (normal)"}
- **Stage Runtime Budget**: {runtime_config.get('max_stage_runtime_minutes', 60)} minutes

### Hardware Configuration:
- **CPU Cores**: {hardware_config.get('cpu_cores', 8)}
- **RAM**: {hardware_config.get('ram_gb', 32)} GB
- **GPU Available**: {"Yes" if hardware_config.get('gpu_available', False) else "No"}
"""
    
    if debug_mode:
        context += """
⚠️ **DEBUG MODE ACTIVE**:
- Use reduced resolution (multiply base resolution by resolution factor)
- Minimize simulation complexity
- Focus on quick validation (~5 min), not accuracy
"""
    
    return context
```

**Which agents receive config?**

| Agent | `runtime_config` | `hardware_config` | Why |
|-------|------------------|-------------------|-----|
| SimulationDesignerAgent | ✓ | ✓ | Needs debug mode, hardware for runtime estimates |
| CodeGeneratorAgent | ✓ | ✓ | Parallelization decisions, memory limits |
| PlannerAgent | ✗ | ✗ | Plans based on paper, not machine |
| ResultsAnalyzerAgent | ✗ | ✗ | Analyzes outputs, doesn't need runtime info |

## Agent Context Details

### PlannerAgent

**Receives**: Full paper information for comprehensive planning.

```
═══════════════════════════════════════════════════════════════════════
CURRENT PAPER (injected at runtime)
═══════════════════════════════════════════════════════════════════════

**Paper ID**: aluminum_nanoantenna_2013
**Title**: Aluminum nanoantenna complexes...
**Domain**: plasmonics

### Paper Text:
[Full extracted text from PDF]

### Figures to Reproduce (4 total):
- **Fig3a** [digitized: ✓]: Transmission spectra
- **Fig3b** [digitized: ✗]: Coated disk spectra
```

**Does NOT receive**: Any simulation outputs, code, progress.

---

### SimulationDesignerAgent

**Receives**: Current stage requirements, assumptions, previous feedback, **runtime and hardware configuration**.

```
═══════════════════════════════════════════════════════════════════════
CURRENT TASK (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Runtime Configuration:
- **Debug Mode**: DISABLED (or ENABLED ⚡)
- **Resolution Factor**: 1.0 (or 0.5x in debug mode)
- **Stage Runtime Budget**: 15 minutes

### Hardware Configuration:
- **CPU Cores**: 8
- **RAM**: 32 GB
- **GPU Available**: No

### Stage Information:
- **Stage ID**: stage1_single_disk
- **Stage Type**: SINGLE_STRUCTURE
- **Design Revision**: 1 of 3

### Stage Requirements:
{stage definition from plan}

### Current Assumptions:
**Global Assumptions:**
*Materials*:
  - Using Palik Al data (n,k from 400-800nm)
  - J-aggregate Lorentzian: ω₀=2.1eV, γ=50meV

### Previous Feedback (if revising):
N/A - first design attempt
```

**Debug Mode Behavior**: When debug mode is enabled, SimulationDesignerAgent should:
- Use reduced resolution (multiply by `debug_resolution_factor`)
- Minimize simulation complexity
- Target quick validation (~5 min) over accuracy

**Does NOT receive**: Full paper text, other stages, code.

---

### CodeGeneratorAgent

**Receives**: Approved design, performance estimate, feedback if revising.

```
═══════════════════════════════════════════════════════════════════════
CURRENT TASK (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Design to Implement:
[Design specification from SimulationDesignerAgent]

### Previous Feedback (if revising):
N/A - first code attempt

### Performance Estimate:
{
  "estimated_runtime_minutes": 5,
  "estimated_memory_gb": 2.0
}

### Output Configuration:
- **Paper ID**: aluminum_nanoantenna_2013
- **Stage ID**: stage1_single_disk
```

**Does NOT receive**: Full paper, assumptions, figures.

---

### CodeReviewerAgent

**Receives**: Design OR code to review (mode-dependent), stage requirements.

**Design Review Mode**:
```
### Review Type: DESIGN REVIEW

### Stage Information:
- **Stage ID**: stage1_single_disk
- **Stage Type**: SINGLE_STRUCTURE

### Design to Review:
[Full design specification]
```

**Code Review Mode**:
```
### Review Type: CODE REVIEW

### Code to Review:
```python
[Full simulation code]
```

### What the Code Should Implement:
[Design summary - truncated to 500 chars]
```

**Does NOT receive**: Full paper, figures, previous outputs.

---

### ExecutionValidatorAgent

**Receives**: Execution results only - stdout, stderr, files, exit code.

```
═══════════════════════════════════════════════════════════════════════
EXECUTION VALIDATION TASK (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Execution Results:
- **Exit Code**: 0
- **Runtime**: 127.3 seconds

### Output Files Created:
- stage1_transmission.csv
- stage1_spectrum.png

### Stdout:
[Last 100 lines]

### Stderr:
[Full stderr if any]
```

**Does NOT receive**: Code, design, paper, figures.

---

### PhysicsSanityAgent

**Receives**: Data files, simulation code (for context), stage type.

```
═══════════════════════════════════════════════════════════════════════
PHYSICS VALIDATION TASK (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Stage Information:
- **Stage ID**: stage1_single_disk
- **Stage Type**: SINGLE_STRUCTURE
- **Domain**: plasmonics

### Data Files to Validate:
- stage1_transmission.csv
- stage1_field.h5

### Simulation Code (for context):
```python
[Code truncated to 2000 chars]
```

### What Physics to Expect:
[Stage requirements from plan]
```

**Does NOT receive**: Paper text, figures, previous stages' data.

---

### ResultsAnalyzerAgent

**Receives**: Outputs, target figures only, digitized data paths.

```
═══════════════════════════════════════════════════════════════════════
ANALYSIS TASK (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Simulation Outputs:
**Exit Code**: 0
**Runtime**: 127.3 seconds

**Output Files**:
- stage1_transmission.csv
- stage1_spectrum.png

**Stdout** (last 50 lines):
[Truncated output]

### Target Figures to Compare:
- **Fig3a**: Transmission spectra for bare disk
  Image: papers/fig3a.png
  Digitized: papers/fig3a_digitized.csv

### Digitized Data Available:
- Fig3a: papers/fig3a_digitized.csv
```

**Does NOT receive**: Full paper text, non-target figures, code.

---

### ComparisonValidatorAgent

**Receives**: Analyzer's comparisons, analysis summary, target figures.

```
═══════════════════════════════════════════════════════════════════════
COMPARISON VALIDATION TASK (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Analysis Summary:
[Summary from ResultsAnalyzerAgent]

### Figure Comparisons to Validate:
**Fig3a**:
- Classification: partial
- Confidence: 85%
- Comparison Table: [structured data]

### Target Figures (reference):
[Figure details]
```

**Does NOT receive**: Paper text, code, design.

---

### SupervisorAgent

**Receives**: Comprehensive summary (but not raw data).

```
═══════════════════════════════════════════════════════════════════════
CURRENT STATE (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Overall Progress:
- stage0_materials: completed_success (revisions: 1)
- stage1_single_disk: completed_partial (revisions: 2)

### Validation Hierarchy Status:
{
  "material_validation": "passed",
  "single_structure": "partial",
  "arrays_systems": "not_done",
  "parameter_sweeps": "not_done"
}

### Recent Figure Comparisons:
- Fig3a: partial (confidence: 85%)

### Runtime Budget:
- **Remaining**: 95.2 minutes
- **Total Used**: 24.8 minutes

### Pending User Questions:
None
```

**Does NOT receive**: Full paper text, raw code, raw data files.

---

### PromptAdaptorAgent

**Receives**: Truncated paper (for quick domain analysis).

```
═══════════════════════════════════════════════════════════════════════
PAPER SUMMARY (for prompt adaptation analysis)
═══════════════════════════════════════════════════════════════════════

**Paper ID**: aluminum_nanoantenna_2013
**Title**: Aluminum nanoantenna complexes...
**Declared Domain**: plasmonics

### Paper Text (first 5000 chars for analysis):
[Truncated paper text]

[... truncated for initial analysis ...]

### Available Agents to Adapt:
- prompt_adaptor
- planner
- simulation_designer
- code_generator
- code_reviewer
- execution_validator
- physics_sanity
- results_analyzer
- comparison_validator
- supervisor
```

**Does NOT receive**: Full paper, figures, any simulation state.

---

## Utility Functions

### Preview Thresholds Table

```python
from src.prompts import preview_injected_thresholds

print(preview_injected_thresholds())
# | Quantity | Excellent | Acceptable | Investigate |
# |----------|-----------|------------|-------------|
# | Resonance wavelength | ±2% | ±5% | >10% |
# | ...
```

### Find Placeholders in Prompts

```python
from src.prompts import list_placeholders_in_prompts

placeholders = list_placeholders_in_prompts()
# {
#   "planner_agent.md": ["{MAX_REPLANS}"],
#   "global_rules.md": ["{THRESHOLDS_TABLE}"],
#   ...
# }
```

### Test Prompt Building

```bash
python -m src.prompts
```

This runs a self-test that:
1. Previews the thresholds table
2. Lists all placeholders found in prompts
3. Builds a test prompt and shows the first 500 chars

## Design Principles

### Principle of Least Context

Each agent receives **only what it needs**:

| Agent | Context Size | Why |
|-------|--------------|-----|
| PlannerAgent | Large (full paper) | Needs complete information for planning |
| CodeGeneratorAgent | Small (design only) | Design contains everything needed |
| ExecutionValidatorAgent | Minimal (outputs only) | Only checks execution success |
| SupervisorAgent | Summary (no raw data) | Strategic decisions need overview |

**Benefits**:
- Prevents context bloat (LLM context windows are limited)
- Faster inference (less context = faster response)
- Lower cost (fewer tokens)
- Easier debugging (each call is self-contained)

### Single Source of Truth

Constants are defined once in `schemas/state.py`:

```python
# schemas/state.py (canonical)
DISCREPANCY_THRESHOLDS = {
    "resonance_wavelength": {"excellent": 2, "acceptable": 5, "investigate": 10},
    ...
}

# Agent prompts use placeholders
# prompts/results_analyzer_agent.md
# "Use the following thresholds: {THRESHOLDS_TABLE}"

# At runtime, placeholder is replaced with actual table
```

### No Conversation History Between Agents

Agents don't share conversation history. Information flows via **state fields**:

```
CodeGeneratorAgent (attempt 1)
    ↓ generates code
CodeReviewerAgent
    ↓ finds issues
    ↓ writes to state["reviewer_feedback"]
CodeGeneratorAgent (attempt 2)
    ↓ receives fresh prompt + state["reviewer_feedback"]
    ↓ uses feedback to improve
```

## Adding New Agents

To add context injection for a new agent:

1. **Add context builder function** in `src/prompts.py`:

```python
def _build_new_agent_context(state: Dict[str, Any]) -> str:
    """Build context section for NewAgent."""
    return f"""
═══════════════════════════════════════════════════════════════════════
TASK FOR NEW AGENT (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Relevant Information:
{state.get('relevant_field', 'default')}
"""
```

2. **Register in inject_state_context**:

```python
elif agent_name == "new_agent":
    context_section = _build_new_agent_context(state)
```

3. **Add to AGENT_PROMPTS dict**:

```python
AGENT_PROMPTS = {
    ...
    "new_agent": "new_agent.md",
}
```

4. **Create the prompt file** in `prompts/new_agent.md`

## Troubleshooting

### Placeholder Not Replaced

If you see `{SOMETHING}` in the final prompt:
1. Check if placeholder is in `inject_constants()`
2. Verify spelling matches exactly
3. Run `list_placeholders_in_prompts()` to find all placeholders

### Context Not Appearing

If agent doesn't receive expected context:
1. Check `inject_state_context()` has case for your agent
2. Verify agent name matches exactly (use `AGENT_PROMPTS.keys()`)
3. Add debug logging to context builder

### Prompt Too Long

If prompt exceeds context window:
1. Check if context builder is truncating properly
2. Consider summarizing large state fields
3. For very long papers, PlannerAgent may need chunking (future enhancement)


