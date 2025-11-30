# BiroClick: Multi-Agent Paper Reproduction System

A LangGraph-based multi-agent system that automatically reproduces simulation results from optics and metamaterials research papers using Meep FDTD simulations.

## Overview

BiroClick reads scientific papers, plans staged reproductions, generates and runs Meep simulations, and systematically compares results to published figures—all while maintaining transparency through explicit assumption tracking and structured progress logging.

### Key Features

- **Stage-based reproduction**: Mandatory validation hierarchy (materials → single structure → arrays → sweeps)
- **Transparent assumptions**: Every inferred parameter is documented with source and reasoning
- **Performance-aware**: Designed for laptop execution with runtime budgets
- **Quantitative tracking**: Structured discrepancy logging with acceptance thresholds
- **Scientific rigor**: Multi-agent review system with Critic and Supervisor oversight

## Architecture

### Agents

| Agent | Role | Responsibilities |
|-------|------|------------------|
| **PlannerAgent** | Strategic planning | Reads paper, extracts parameters, classifies figures, designs staged reproduction plan |
| **ExecutorAgent** | Implementation | Designs simulations, writes Meep code, analyzes results, documents discrepancies |
| **CodeReviewerAgent** | Pre-run QA | Reviews code, geometry, materials, numerics before simulation runs |
| **ResultsValidatorAgent** | Post-run QA | Validates outputs, physics, figure comparisons after simulation runs |
| **SupervisorAgent** | Scientific oversight | Big-picture assessment, validation hierarchy monitoring, decision-making |

### Workflow

```
START
  ↓
PLAN (PlannerAgent)
  ↓
SELECT_STAGE
  ├→ [no more stages] → END
  └→ [has next stage] → DESIGN (ExecutorAgent)
        ↓
     CODE_REVIEW (CodeReviewerAgent)
        ├→ [needs_revision] → DESIGN (max 3 times)
        └→ [approve_to_run] → RUN_CODE
              ↓
           ANALYZE (ExecutorAgent)
              ↓
           VALIDATE_RESULTS (ResultsValidatorAgent)
              ├→ [needs_revision] → ANALYZE (max 2 times)
              └→ [approve_results] → SUPERVISOR
                    ├→ [ok_continue] → SELECT_STAGE
                    ├→ [replan_needed] → PLAN
                    └→ [ask_user] → USER_INPUT
```

### Mandatory Staging Order

Every reproduction follows this validation hierarchy:

1. **Stage 0: Material Validation** — Verify optical constants against paper data
2. **Stage 1: Single Structure** — Isolated structure validation (no arrays)
3. **Stage 2+: Arrays/Systems** — Add periodicity and collective effects
4. **Parameter Sweeps** — Reproduce multi-parameter figures
5. **Complex Physics** — Nonlinear, emission, Purcell (only after linear validation)

**Each stage must pass before proceeding.** Early failures compound.

## Project Structure

```
biroclick/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── prompts/                       # Agent system prompts
│   ├── global_rules.md            # Non-negotiable rules for all agents
│   ├── planner_agent.md           # PlannerAgent system prompt
│   ├── executor_agent.md          # ExecutorAgent system prompt
│   ├── code_reviewer_agent.md     # CodeReviewerAgent (pre-run QA)
│   ├── results_validator_agent.md # ResultsValidatorAgent (post-run QA)
│   ├── supervisor_agent.md        # SupervisorAgent system prompt
│   └── report_template.md         # REPRODUCTION_REPORT.md template
│
├── schemas/                  # Data model definitions
│   ├── plan_schema.json      # Plan file structure with example
│   ├── assumptions_schema.json # Assumptions tracking schema
│   ├── progress_schema.json  # Progress logging schema
│   ├── report_schema.json    # Reproduction report schema
│   └── state.py              # Python TypedDict for LangGraph state
│
├── docs/                     # Additional documentation
│   ├── workflow.md           # Detailed workflow documentation
│   └── guidelines.md         # Optics reproduction guidelines
│
└── src/                      # Source code (implementation)
    ├── __init__.py
    ├── agents.py             # Agent node implementations
    ├── graph.py              # LangGraph state graph definition
    ├── utils.py              # Utility functions
    └── code_runner.py        # Sandboxed Python execution
```

## Data Models

### Plan (`plan_<paper_id>.json`)

Defines the reproduction strategy:
- Extracted parameters with sources (text/figure/supplementary)
- Target figures with simulation classification
- Staged reproduction plan with dependencies and runtime budgets

### Assumptions (`assumptions_<paper_id>.json`)

Tracks all non-explicit parameters:
- Global assumptions (materials, boundary conditions)
- Geometry interpretations (spacing vs period, conformal layers)
- Stage-specific assumptions
- Validation status for each assumption

### Progress (`progress_<paper_id>.json`)

Logs reproduction progress:
- Stage status (not_started → in_progress → completed_*)
- Output files with result classifications
- Structured discrepancy tracking
- Issues and next actions

### Reproduction Report (`REPRODUCTION_REPORT_<paper_id>.md`)

Final output document with:
- **Executive Summary**: Overall assessment table with status icons
- **Simulation Assumptions**: Three tables (direct, interpreted, implementation)
- **Figure Comparisons**: Side-by-side images with feature/shape comparison tables
- **Summary Table**: All figures at a glance
- **Systematic Discrepancies**: Named issues affecting multiple results
- **Conclusions**: Key findings and final statement

## Quantitative Thresholds

Reproduction quality is assessed using standardized thresholds:

| Quantity | Excellent | Acceptable | Investigate |
|----------|-----------|------------|-------------|
| Resonance wavelength | ±2% | ±5% | >10% |
| Linewidth / Q-factor | ±10% | ±30% | >50% |
| Transmission/reflection | ±5% | ±15% | >30% |
| Field enhancement | ±20% | ±50% | >2× |

## Key Principles

### 1. Figures Over Text
When paper text and figures disagree, **figures are more reliable**. Text may have typos; figures show actual data.

### 2. Explicit Assumptions
Every inferred value must be documented:
- What was assumed
- Why it's reasonable
- What alternatives exist
- Whether it's been validated

### 3. Honest Assessment
A documented partial reproduction with understood limitations is **more valuable** than an undocumented "perfect" match.

### 4. When to Stop
Stop optimizing when:
- Main physics phenomenon is qualitatively visible
- Quantitative agreement is within acceptable ranges
- Remaining discrepancies are understood
- Further improvement requires unavailable information

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/biroclick.git
cd biroclick

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-api-key"
```

## Usage

```python
from src.graph import create_repro_graph

# Initialize the graph
app = create_repro_graph()

# Run reproduction
initial_state = {
    "paper_id": "aluminum_nanoantenna_2013",
    "paper_text": "... extracted paper text ...",
    "paper_domain": "plasmonics"
}

result = app.invoke(initial_state)
```

## Dependencies

- **langgraph**: Multi-agent orchestration
- **langchain-openai**: LLM integration
- **meep**: FDTD simulations
- **numpy**: Numerical computing
- **matplotlib**: Plotting
- **h5py**: HDF5 data storage

## Contributing

Contributions are welcome! Please read the guidelines in `docs/guidelines.md` for lessons learned from optics paper reproduction.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This project builds on lessons learned from systematic paper reproduction efforts in optics and photonics.
