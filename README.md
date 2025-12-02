# ReproLab: Multi-Agent Paper Reproduction System

**Status**: v0.1.0 (Planning + Foundation)

A LangGraph-based multi-agent system that automatically reproduces simulation results from optics and metamaterials research papers using Meep FDTD simulations.

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| **Core State & Schemas** | Implemented | `schemas/state.py`, `schemas/*.json` |
| **Paper Loader** | Implemented | `src/paper_loader.py` |
| **Code Runner (sandboxed execution)** | Implemented | `src/code_runner.py` |
| **Graph Wiring** | Skeleton only | `src/graph.py` |
| **Agent Node Implementations** | Stubs only | `src/agents.py` |
| **Prompt Templates** | Complete | `prompts/*.md` |
| **Material Database** | Complete | `materials/` |

**Note:** Code snippets in `docs/workflow.md` and `docs/guidelines.md` are illustrative design specifications, not guaranteed to match current implementation.

## MVP Scope (v0)

The first implementation targets a minimal vertical slice to validate the architecture:

**Included in v0:**
- Single test paper (Stage 0 material validation + one Stage 1 single structure)
- Core agent path: Planner → PlanReviewer → Designer → DesignReviewer → CodeGenerator → CodeReviewer → RUN_CODE → ExecutionValidator → ResultsAnalyzer → Supervisor
- **Dedicated review nodes**: Separate PlanReviewer, DesignReviewer, and CodeReviewer agents
- **Mandatory material checkpoint**: Explicit `material_checkpoint` node after Stage 0 requiring user confirmation
- **Cross-stage backtracking**: `handle_backtrack` node for invalidating and re-running stages
- **Single source of truth**: Validation hierarchy computed from progress (not stored separately)
- **Context management**: `context_budget` tracking implemented with automatic token estimation and recovery actions
- **Metrics pipeline**: Token usage tracking via `metrics` field in state
- **Physics-driven redesign**: `design_flaw` verdict routes physics failures to design (not code)
- **Validated materials handoff**: `validated_materials` field passed from Stage 0 to all subsequent stages
- **Output artifact specs**: `expected_outputs` in stage schema for consistent file naming
- **Digitized data enforcement**: Targets requiring <2% precision must have digitized data paths
- Basic error handling (max 3 revisions, then escalate)
- Simple report generation

**Out of scope for v0 (true deferrals):**
- Multi-paper batch processing
- Prompt adaptation agent customizations (PromptAdaptorAgent is stub-only)
- Parameter sweep stages (only Stage 0 + Stage 1 single structure)
- Advanced comparison validation refinements

## Overview

ReproLab reads scientific papers, plans staged reproductions, generates and runs Meep simulations, and systematically compares results to published figures—all while maintaining transparency through explicit assumption tracking and structured progress logging.

### Key Features

- **Stage-based reproduction**: Adaptive validation hierarchy (materials always first → paper-dependent stages)
- **Transparent assumptions**: Every inferred parameter is documented with source and reasoning
- **Performance-aware**: Designed for laptop execution with runtime budgets
- **Quantitative tracking**: Structured discrepancy logging with acceptance thresholds
- **Scientific rigor**: Multi-agent review system with specialized validators
- **Multimodal AI**: Uses Claude Opus 4.5 for all agents (vision-capable for figure comparison)

### How This Differs From Typical LangGraph Demos

Most LangGraph tutorials show chatbots or simple pipelines. ReproLab is designed for **scientific reproducibility**, which requires different patterns:

| Typical Demo | ReproLab |
|--------------|----------|
| Free-form dict state | Heavily typed `TypedDict` with validation |
| Linear or simple branching | Strict validation hierarchy (can't run Stage 2 until Stage 1 passes) |
| Trust LLM output | Multi-agent review before every execution |
| Implicit assumptions | Every inferred parameter documented with source and reasoning |
| Run until done | Runtime budgets, revision limits, mandatory user checkpoints |
| Success/failure binary | Quantitative thresholds (excellent/acceptable/investigate) |

This isn't "just another agents demo"—it's a framework for rigorous, auditable computational science.

## Architecture

### Agents (13 total)

| Agent | Role | Responsibilities |
|-------|------|------------------|
| **PromptAdaptorAgent** | System customization | Analyzes paper, adapts agent prompts for domain-specific needs |
| **PlannerAgent** | Strategic planning | Reads paper, extracts parameters, classifies figures, designs staged reproduction plan |
| **PlanReviewerAgent** | Plan QA | Reviews reproduction plan before stage execution begins |
| **SimulationDesignerAgent** | Simulation design | Interprets geometry, selects materials, designs sources/BCs, estimates performance |
| **DesignReviewerAgent** | Design QA | Reviews simulation design before code generation |
| **CodeGeneratorAgent** | Code generation | Writes Python+Meep code from approved designs |
| **CodeReviewerAgent** | Code QA | Reviews generated code before execution (Meep API, unit normalization, etc.) |
| **ExecutionValidatorAgent** | Execution validation | Validates simulation ran correctly, checks output files |
| **PhysicsSanityAgent** | Physics validation | Validates conservation laws, value ranges, numerical quality |
| **ResultsAnalyzerAgent** | Analysis | Compares results to paper, classifies success/partial/failure |
| **ComparisonValidatorAgent** | Comparison QA | Validates comparison accuracy, math, and classifications |
| **SupervisorAgent** | Scientific oversight | Big-picture assessment, validation hierarchy monitoring, decision-making |
| **ReportGeneratorAgent** | Report generation | Synthesizes all stage results into final reproduction report |

### Workflow

```
START
  ↓
ADAPT_PROMPTS (PromptAdaptorAgent) ← Customizes system for paper
  ↓
PLAN (PlannerAgent) ← Uses adapted prompts
  ↓
PLAN_REVIEW (PlanReviewerAgent) ← reviews plan
  ├→ [needs_revision] → PLAN (max 2 replans)
  └→ [approve] ↓
        ↓
SELECT_STAGE
  ├→ [no more stages] → GENERATE_REPORT → END
  └→ [has next stage] ↓
        ↓
     DESIGN (SimulationDesignerAgent)
        ↓
     DESIGN_REVIEW (DesignReviewerAgent) ← reviews design
        ├→ [needs_revision] → DESIGN (max 3 times)
        └→ [approve] → GENERATE_CODE (CodeGeneratorAgent)
              ↓
           CODE_REVIEW (CodeReviewerAgent) ← reviews code only
              ├→ [needs_revision] → GENERATE_CODE (max 3 times)
              └→ [approve] → RUN_CODE
                    ↓
                 EXECUTION_CHECK (ExecutionValidatorAgent)
                    ├→ [fail] → GENERATE_CODE
                    └→ [pass] → PHYSICS_CHECK (PhysicsSanityAgent)
                          ├→ [fail] → GENERATE_CODE
                          ├→ [design_flaw] → DESIGN ← Physics issues requiring redesign
                          └→ [pass] → ANALYZE (ResultsAnalyzerAgent)
                                ↓
                             COMPARISON_CHECK (ComparisonValidatorAgent)
                                ├→ [needs_revision] → ANALYZE (max 2 times)
                                └→ [approve] → SUPERVISOR
                                      ├→ [ok_continue + Stage 0] → MATERIAL_CHECKPOINT → ASK_USER
                                      ├→ [ok_continue + other] → SELECT_STAGE
                                      ├→ [backtrack_to_stage] → HANDLE_BACKTRACK → SELECT_STAGE
                                      ├→ [replan_needed] → PLAN
                                      └→ [ask_user] → ASK_USER → SUPERVISOR
```

**Key features:**
- **Separate review nodes**: Plan, Design, and Code each have dedicated reviewers
- **Material checkpoint**: After Stage 0 completes, `material_checkpoint` node ALWAYS routes to `ask_user` for mandatory user confirmation
- **Backtracking**: `handle_backtrack` node marks target stage as `needs_rerun` and dependent stages as `invalidated`

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
│   ├── plan_reviewer_agent.md            # PlanReviewerAgent prompt
│   ├── simulation_designer_agent.md      # SimulationDesignerAgent prompt
│   ├── design_reviewer_agent.md          # DesignReviewerAgent prompt
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
│   ├── metrics_schema.json   # Metrics tracking schema
│   ├── report_schema.json    # Reproduction report schema
│   ├── prompt_adaptations_schema.json # Prompt adaptation schema
│   ├── *_output_schema.json  # Agent output schemas for function calling
│   └── state.py              # LangGraph workflow state (imports generated types)
│
├── docs/                     # Additional documentation
│   ├── workflow.md           # Detailed workflow documentation
│   ├── guidelines.md         # Optics reproduction guidelines
│   ├── preparing_papers.md   # Paper preparation & ingestion guide
│   └── prompt_building.md    # Agent prompt construction reference
│
├── outputs/                  # Simulation outputs (gitignored except .gitkeep)
├── papers/                   # Paper markdown and figures (gitignored except .gitkeep)
├── tests/                    # Unit tests
│
└── src/                      # Source code (implementation)
    ├── __init__.py
    ├── paper_loader.py       # Paper input schema and validation
    ├── agents.py             # Agent node implementations (stubs)
    ├── graph.py              # LangGraph state graph definition
    ├── prompts.py            # Prompt building and context injection
    └── code_runner.py        # Sandboxed Python execution
```

## Paper Input

The system uses **Claude Opus 4.5** (multimodal) to analyze both text and images. Paper inputs are provided via a structured format:

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

> **Note**: Artifact files use `_artifact_` prefix (e.g., `_artifact_plan.json`) to indicate
> they are for human review only, not for execution. See [docs/workflow.md](docs/workflow.md)
> for the dual checkpointing strategy.

### Plan (`_artifact_plan.json`)

Defines the reproduction strategy:
- Extracted parameters with sources (text/figure/supplementary)
- Target figures with simulation classification
- Staged reproduction plan with dependencies and runtime budgets

### Assumptions (`_artifact_assumptions.json`)

Tracks all non-explicit parameters:
- Global assumptions (materials, boundary conditions)
- Geometry interpretations (spacing vs period, conformal layers)
- Stage-specific assumptions
- Validation status for each assumption

### Progress (`_artifact_progress.json`)

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

### Type Generation from JSON Schemas

**JSON schemas are the source of truth** for data types. Python TypedDicts should be generated from them to ensure consistency.

```bash
# Install generator
pip install datamodel-code-generator

# Generate Python types from all schemas
datamodel-codegen \
    --input schemas/plan_schema.json \
    --input schemas/progress_schema.json \
    --input schemas/metrics_schema.json \
    --input schemas/report_schema.json \
    --input-file-type jsonschema \
    --output-model-type typing.TypedDict \
    --output schemas/generated_types.py
```

The `schemas/state.py` file contains workflow-specific types (`ReproState`, `RuntimeConfig`, etc.) and imports the generated types for schema-defined structures.

> **Note**: Re-run generation whenever JSON schemas are modified.

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

### Platform Support

| Platform | Support Level | Notes |
|----------|--------------|-------|
| **Linux** | ✅ Full | Primary development platform |
| **macOS** | ✅ Full | Intel and Apple Silicon |
| **Windows + WSL2** | ✅ Full | Recommended for Windows users |
| **Windows Native** | ❌ Not Recommended | No memory limits, use WSL2 instead |

**Windows Users**: We recommend using [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) for full functionality. Native Windows has limited resource management. See `docs/guidelines.md` Section 14 for details.

### Quick Start (Linux/macOS/WSL2)

```bash
# Clone the repository
git clone https://github.com/yourusername/reprolab.git
cd reprolab

# Create conda environment (recommended - required for Meep)
conda create -n reprolab python=3.11
conda activate reprolab

# Install Meep via conda (REQUIRED - do not use pip)
conda install -c conda-forge meep=1.28.0

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
export ANTHROPIC_API_KEY="your-api-key"

# Verify installation
python -c "import meep; print(f'Meep {meep.__version__} installed successfully')"
```

### Development Installation

```bash
# After the quick start above, also install dev dependencies
pip install -r requirements-dev.txt

# Generate Python types from JSON schemas (one-time setup)
python scripts/generate_types.py

# Run tests
pytest tests/

# Type checking
mypy src/
```

> **Note:** Re-run `python scripts/generate_types.py` whenever JSON schemas in `schemas/` are modified.
> This generates `schemas/generated_types.py` which provides TypedDict types matching the schemas.

## Usage

```python
from src.graph import create_repro_graph
from src.paper_loader import create_paper_input, create_state_from_paper_input

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

# Convert paper input to initial state (handles field mapping: figures → paper_figures)
initial_state = create_state_from_paper_input(paper_input)

# Run reproduction
result = app.invoke(initial_state)
```

### Quick Debug Mode

Don't want to commit to a full multi-hour run? Use debug mode for fast sanity checks:

```python
from src.graph import create_repro_graph
from schemas.state import DEBUG_RUNTIME_CONFIG
from src.paper_loader import create_state_from_paper_input

# Debug mode: 30 min max, minimal stages, lower resolution
app = create_repro_graph()
initial_state = create_state_from_paper_input(
    paper_input, 
    runtime_config=DEBUG_RUNTIME_CONFIG
)
result = app.invoke(initial_state)

# Check diagnostic summary
print(result.get("debug_summary"))
```

Debug mode runs Stage 0 (materials) + minimal Stage 1 only, with reduced resolution. Use it to verify paper loading, material models, and basic geometry before committing to a full reproduction. See `docs/workflow.md` for detailed debug mode documentation.

## Security Notice ⚠️

**v1 sandboxing is suitable for local use with trusted inputs only.**

The code execution sandbox (`src/code_runner.py`) uses subprocess isolation with:
- Timeout limits
- Memory limits (Unix only, via `resource` module)
- Text-based pattern detection for dangerous operations

**Known Limitations**:
- Python is dynamic; determined/malicious code can bypass text-based checks:
  ```python
  # These patterns are NOT detected by simple regex:
  getattr(__import__('os'), 'sys'+'tem')('dangerous')
  exec(__import__('base64').b64decode('...'))
  ```
- No network isolation (simulations shouldn't need network, but it's not enforced)
- LLM-generated code runs with your user privileges
- Windows has no memory limiting (only timeout)

**Recommendations**:
| Use Case | Recommendation |
|----------|---------------|
| Local development | Current sandbox is acceptable |
| Shared server | Use Docker/Podman container isolation |
| Production/hosted | **Mandatory**: Container with ephemeral filesystem, no network |

**Future (v2)**: Full container-based isolation with network disabled at container level.

## Current Limitations (v0)

These are implementation quality limitations, not missing features. All features in "MVP Scope (v0)" are fully designed and wired into the system:

| Limitation | Description | Future Plan |
|------------|-------------|-------------|
| **PDF conversion required** | PDFs must be converted to markdown first (using marker/nougat) | Direct PDF loading via `load_paper_from_pdf()` |
| **Manual figure digitization** | Reference data requires manual digitization with tools like WebPlotDigitizer | Integration with automatic digitizers |
| **Figure download from markdown** | ✅ `load_paper_from_markdown()` extracts and downloads figures automatically | Planned for v1 |
| **Supplementary materials** | ✅ Supported via `supplementary_markdown_path` parameter | Planned for v1 |
| **Long paper handling** | ✅ Warnings displayed for papers >50K chars; manual trimming recommended | Smart chunking/summarization |
| **Single-model decisions** | Each agent uses one LLM model | Parallel multi-model consensus for critics/supervisors/planners |
| **Single-threaded execution** | Stages run sequentially, not in parallel | Stage parallelization where dependencies allow |
| **Optics/photonics focus** | Primarily tested on plasmonics/metamaterials papers | Domain expansion to other physics areas |
| **Meep-only simulations** | Only supports Meep FDTD backend | Multi-backend support (COMSOL, Lumerical) |
| **Basic sandboxing** | Subprocess isolation with timeout/memory limits | Full container isolation (Docker) for production |

See `docs/guidelines.md` Section 14 for future improvement roadmap.

## Testing

### Schema-Type Sync Tests

To verify JSON schemas stay synchronized with Python types in `state.py`:

```bash
python tests/test_schema_type_sync.py
```

These tests catch drift between:
- JSON schemas (source of truth for data structures)
- Python TypedDicts in `state.py` (used at runtime)
- Agent output schema required fields

If tests fail, it indicates `state.py` or schemas need to be updated to match.

## Dependencies

- **langgraph**: Multi-agent orchestration
- **langchain-anthropic**: Claude Opus 4.5 integration (primary LLM)
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
