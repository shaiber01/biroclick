# ReproLab: Multi-Agent Paper Reproduction System

A LangGraph-based multi-agent system that automatically reproduces simulation results from optics and metamaterials research papers using Meep FDTD simulations.

## Overview

ReproLab reads scientific papers, plans staged reproductions, generates and runs Meep simulations, and systematically compares results to published figures—all while maintaining transparency through explicit assumption tracking and structured progress logging.

### Key Features

- **Stage-based reproduction**: Adaptive validation hierarchy (materials always first → paper-dependent stages)
- **Transparent assumptions**: Every inferred parameter is documented with source and reasoning
- **Performance-aware**: Designed for laptop execution with runtime budgets
- **Quantitative tracking**: Structured discrepancy logging with acceptance thresholds
- **Scientific rigor**: Multi-agent review system with specialized validators
- **Multimodal AI**: Uses vision-capable LLMs (GPT-4o, Claude) for figure comparison

## Architecture

### Agents (10 total)

| Agent | Role | Responsibilities |
|-------|------|------------------|
| **PromptAdaptorAgent** | System customization | Analyzes paper, adapts agent prompts for domain-specific needs |
| **PlannerAgent** | Strategic planning | Reads paper, extracts parameters, classifies figures, designs staged reproduction plan |
| **SimulationDesignerAgent** | Simulation design | Interprets geometry, selects materials, designs sources/BCs, estimates performance |
| **CodeGeneratorAgent** | Code generation | Writes Python+Meep code from approved designs |
| **CodeReviewerAgent** | Pre-run QA | Reviews designs and code before execution |
| **ExecutionValidatorAgent** | Execution validation | Validates simulation ran correctly, checks output files |
| **PhysicsSanityAgent** | Physics validation | Validates conservation laws, value ranges, numerical quality |
| **ResultsAnalyzerAgent** | Analysis | Compares results to paper, classifies success/partial/failure |
| **ComparisonValidatorAgent** | Comparison QA | Validates comparison accuracy, math, and classifications |
| **SupervisorAgent** | Scientific oversight | Big-picture assessment, validation hierarchy monitoring, decision-making |

### Workflow

```
START
  ↓
ADAPT_PROMPTS (PromptAdaptorAgent) ← Customizes system for paper
  ↓
PLAN (PlannerAgent) ← Uses adapted prompts
  ↓
SELECT_STAGE
  ├→ [no more stages] → GENERATE_REPORT → END
  └→ [has next stage] ↓
        ↓
     DESIGN (SimulationDesignerAgent)
        ↓
     CODE_REVIEW (CodeReviewerAgent) ← reviews design
        ├→ [needs_revision] → DESIGN (max 3 times)
        └→ [approve] → GENERATE_CODE (CodeGeneratorAgent)
              ↓
           CODE_REVIEW (CodeReviewerAgent) ← reviews code
              ├→ [needs_revision] → GENERATE_CODE (max 3 times)
              └→ [approve] → RUN_CODE
                    ↓
                 EXECUTION_CHECK (ExecutionValidatorAgent)
                    ├→ [fail] → GENERATE_CODE
                    └→ [pass] → PHYSICS_CHECK (PhysicsSanityAgent)
                          ├→ [fail] → GENERATE_CODE
                          └→ [pass] → ANALYZE (ResultsAnalyzerAgent)
                                ↓
                             COMPARISON_CHECK (ComparisonValidatorAgent)
                                ├→ [needs_revision] → ANALYZE (max 2 times)
                                └→ [approve] → SUPERVISOR
                                      ├→ [ok_continue] → SELECT_STAGE
                                      ├→ [replan_needed] → PLAN
                                      └→ [ask_user] → USER_INPUT
```

### Adaptive Validation Hierarchy

**Core principle**: Validate foundations before adding complexity. Stages adapt to the paper's content.

**Always required**:
- **Stage 0: Material Validation** — Verify optical constants (requires user confirmation)

**Paper-dependent stages** (include only what applies):
- **Single Structure** — When paper shows isolated structures
- **Arrays/Systems** — When paper shows periodic/coupled structures  
- **Parameter Sweeps** — When paper shows multi-parameter data
- **Complex Physics** — When paper involves nonlinear/transient effects

Not all papers need all stages. A single nanoparticle paper may only need Stage 0 → Single Structure.
A photonic crystal paper may skip "single structure" entirely since periodicity IS the structure.

## Project Structure

```
reprolab/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── prompts/                              # Agent system prompts
│   ├── global_rules.md                   # Non-negotiable rules for all agents
│   ├── prompt_adaptor_agent.md           # PromptAdaptorAgent (runs first)
│   ├── planner_agent.md                  # PlannerAgent system prompt
│   ├── simulation_designer_agent.md      # SimulationDesignerAgent prompt
│   ├── code_generator_agent.md           # CodeGeneratorAgent prompt
│   ├── code_reviewer_agent.md            # CodeReviewerAgent prompt
│   ├── execution_validator_agent.md      # ExecutionValidatorAgent prompt
│   ├── physics_sanity_agent.md           # PhysicsSanityAgent prompt
│   ├── results_analyzer_agent.md         # ResultsAnalyzerAgent prompt
│   ├── comparison_validator_agent.md     # ComparisonValidatorAgent prompt
│   ├── supervisor_agent.md               # SupervisorAgent prompt
│   └── report_template.md                # REPRODUCTION_REPORT.md template
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
    ├── paper_loader.py       # Paper input schema and validation
    ├── agents.py             # Agent node implementations
    ├── graph.py              # LangGraph state graph definition
    ├── utils.py              # Utility functions
    └── code_runner.py        # Sandboxed Python execution
```

## Paper Input

The system uses **multimodal LLMs** (GPT-4o, Claude) that can analyze both text and images. Paper inputs are provided via a structured format:

```python
from src.paper_loader import create_paper_input

paper_input = create_paper_input(
    paper_id="aluminum_nanoantenna_2013",
    paper_title="Aluminum nanoantenna complexes for strong coupling...",
    paper_text="... extracted paper text ...",
    figures=[
        {"id": "Fig3a", "description": "Transmission spectra", "image_path": "papers/fig3a.png"},
        {"id": "Fig3b", "description": "Coated disk spectra", "image_path": "papers/fig3b.png"},
    ],
    paper_domain="plasmonics"
)
```

**Figure images** are passed directly to vision-capable LLMs during:
- **PlannerAgent**: Analyzing what figures show and how to reproduce them
- **ResultsAnalyzerAgent**: Comparing simulation outputs to paper figures visually
- **ComparisonValidatorAgent**: Verifying comparison accuracy

See `src/paper_loader.py` for the full `PaperInput` schema and validation.

For detailed instructions on preparing papers (text extraction, figure extraction, digitization), see **[docs/preparing_papers.md](docs/preparing_papers.md)**.

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
git clone https://github.com/yourusername/reprolab.git
cd reprolab

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
from src.paper_loader import create_paper_input

# Initialize the graph
app = create_repro_graph()

# Prepare paper input with figures for multimodal comparison
paper_input = create_paper_input(
    paper_id="aluminum_nanoantenna_2013",
    paper_title="Aluminum nanoantenna complexes for strong coupling...",
    paper_text="... extracted paper text ...",
    figures=[
        {"id": "Fig3a", "description": "Bare disk transmission", "image_path": "papers/fig3a.png"},
        {"id": "Fig3b", "description": "Coated disk transmission", "image_path": "papers/fig3b.png"},
    ],
    paper_domain="plasmonics"
)

# Run reproduction
result = app.invoke(paper_input)
```

## Current Limitations (v1)

This is an early version with several known limitations:

| Limitation | Description | Future Plan |
|------------|-------------|-------------|
| **PDF conversion required** | PDFs must be converted to markdown first (using marker/nougat) | Direct PDF loading via `load_paper_from_pdf()` |
| **Manual figure digitization** | Reference data requires manual digitization with tools like WebPlotDigitizer | Integration with automatic digitizers |
| **Figure download from markdown** | ✅ `load_paper_from_markdown()` extracts and downloads figures automatically | N/A (implemented in v1) |
| **Supplementary materials** | ✅ Supported via `supplementary_markdown_path` parameter | N/A (implemented in v1) |
| **Long paper handling** | ✅ Warnings displayed for papers >50K chars; manual trimming recommended | Smart chunking/summarization |
| **Single-model decisions** | Each agent uses one LLM model | Parallel multi-model consensus for critics/supervisors/planners |
| **Single-threaded execution** | Stages run sequentially, not in parallel | Stage parallelization where dependencies allow |
| **Optics/photonics focus** | Primarily tested on plasmonics/metamaterials papers | Domain expansion to other physics areas |
| **Meep-only simulations** | Only supports Meep FDTD backend | Multi-backend support (COMSOL, Lumerical) |
| **Basic sandboxing** | Subprocess isolation with timeout/memory limits | Full container isolation (Docker) for production |

See `docs/guidelines.md` Section 14 for future improvement roadmap.

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
