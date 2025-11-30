# Workflow Documentation

This document describes the LangGraph workflow for paper reproduction.

## Overview

The system uses a state graph where each node represents an agent action or system operation. State flows through the graph, accumulating results and tracking progress.

## Agent Summary

| Agent | Node | Role |
|-------|------|------|
| PlannerAgent | PLAN | Reads paper, creates staged plan |
| SimulationDesignerAgent | DESIGN | Designs simulation setup |
| CodeReviewerAgent | CODE_REVIEW | Reviews design and code |
| CodeGeneratorAgent | GENERATE_CODE | Writes Python+Meep code |
| ExecutionValidatorAgent | EXECUTION_CHECK | Validates simulation ran correctly |
| ResultsAnalyzerAgent | ANALYZE | Compares results to paper |
| ScientificValidatorAgent | SCIENTIFIC_CHECK | Validates physics and comparisons |
| SupervisorAgent | SUPERVISOR | Big-picture decisions |

## Node Definitions

### 1. PLAN Node (PlannerAgent)

**Purpose**: Analyze paper and create reproduction plan

**Inputs**:
- `paper_id`: Unique identifier
- `paper_text`: Extracted paper content
- `paper_domain`: Field classification

**Outputs**:
- `plan`: Complete staged reproduction plan
- `assumptions`: Initial assumptions with sources
- `progress`: Initialized progress structure
- `extracted_parameters`: Parameters with provenance

**Transitions**:
- → SELECT_STAGE (normal)

---

### 2. SELECT_STAGE Node

**Purpose**: Choose next stage to execute based on dependencies and status

**Logic**:
```python
def select_next_stage(state):
    for stage in state["plan"]["stages"]:
        # Skip completed or blocked stages
        if stage["status"] in ["completed_success", "completed_partial", "blocked"]:
            continue
        
        # Check dependencies
        deps_met = all(
            get_stage_status(dep) in ["completed_success", "completed_partial"]
            for dep in stage["dependencies"]
        )
        
        if deps_met:
            return stage["stage_id"]
    
    return None  # All stages done
```

**Transitions**:
- → DESIGN (has next stage)
- → GENERATE_REPORT (no more stages)

---

### 3. DESIGN Node (SimulationDesignerAgent)

**Purpose**: Design simulation setup for current stage (no code generation)

**Inputs**:
- `current_stage_id`: Stage to implement
- `plan`: Stage definition
- `assumptions`: Current assumptions
- `reviewer_feedback`: Feedback if revising

**Outputs**:
- `design`: Complete simulation design specification
- `performance_estimate`: Runtime/memory estimates
- `new_assumptions`: Any new assumptions introduced
- `output_specifications`: Expected output files and formats

**Transitions**:
- → CODE_REVIEW (always)

---

### 4. CODE_REVIEW Node (CodeReviewerAgent)

**Purpose**: Review design or code before proceeding

**When reviewing DESIGN**:
- [ ] Geometry matches paper interpretation
- [ ] Materials correctly selected with sources
- [ ] Source configuration appropriate
- [ ] Boundary conditions match physics
- [ ] Resolution adequate for features
- [ ] Performance within budget

**When reviewing CODE**:
- [ ] Code implements design correctly
- [ ] Progress prints included
- [ ] No blocking calls (plt.show, input)
- [ ] Error handling present
- [ ] File outputs named correctly

**Outputs**:
- `reviewer_verdict`: "approve" | "needs_revision"
- `reviewer_issues`: List of issues found
- `reviewer_feedback`: Detailed feedback

**Transitions** (after design review):
- → GENERATE_CODE (approved)
- → DESIGN (needs revision, count < 3)
- → ASK_USER (needs revision, count >= 3)

**Transitions** (after code review):
- → RUN_CODE (approved)
- → GENERATE_CODE (needs revision, count < 3)
- → ASK_USER (needs revision, count >= 3)

---

### 5. GENERATE_CODE Node (CodeGeneratorAgent)

**Purpose**: Generate Python+Meep code from approved design

**Inputs**:
- `design`: Approved simulation design from SimulationDesignerAgent
- `reviewer_feedback`: Feedback if revising code

**Outputs**:
- `code`: Complete Python+Meep simulation code
- `expected_outputs`: List of expected output files
- `estimated_runtime_minutes`: Runtime estimate

**Transitions**:
- → CODE_REVIEW (always - code review phase)

---

### 6. RUN_CODE Node (Python Execution)

**Purpose**: Execute the simulation code

**Implementation**:
```python
def run_code_node(state):
    try:
        # Create isolated environment
        env = create_sandbox_env(state["current_stage_id"])
        
        # Execute with timeout
        result = execute_with_timeout(
            state["code"],
            timeout=state["plan"]["stages"][idx]["runtime_budget_minutes"] * 60,
            env=env
        )
        
        return {
            "stage_outputs": {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "files": list_output_files(env),
                "exit_code": result.returncode
            },
            "run_error": None
        }
    
    except TimeoutError:
        return {"run_error": "Simulation exceeded runtime budget"}
    except Exception as e:
        return {"run_error": str(e)}
```

**Transitions**:
- → EXECUTION_CHECK (always)

---

### 7. EXECUTION_CHECK Node (ExecutionValidatorAgent)

**Purpose**: Validate that simulation ran correctly

**Checks**:
- [ ] Simulation completed without errors
- [ ] Exit code was 0
- [ ] All expected output files exist
- [ ] Files are non-empty and valid format
- [ ] No NaN/Inf in data
- [ ] Runtime was reasonable

**Outputs**:
- `execution_verdict`: "pass" | "fail" | "warning"
- `execution_status`: Detailed status object
- `proceed_to_analysis`: Boolean

**Transitions**:
- → ANALYZE (pass or warning)
- → GENERATE_CODE (fail, recoverable error)
- → ASK_USER (fail, unknown error or limit reached)

---

### 8. ANALYZE Node (ResultsAnalyzerAgent)

**Purpose**: Compare results to paper and classify reproduction quality

**Inputs**:
- `stage_outputs`: Files and data from simulation
- `plan`: Target figures and validation criteria
- Paper figures (for comparison)

**Outputs**:
- `per_figure_reports`: Structured comparison for each figure
- `discrepancies`: Documented discrepancies with likely causes
- `progress_update`: Updated stage status

**Per-Result Classification**:
| Classification | Criteria |
|----------------|----------|
| SUCCESS | Qualitative match + within "acceptable" thresholds |
| PARTIAL | Trends match but quantitative in "investigate" range |
| FAILURE | Wrong trends, missing features, or unphysical results |

**Transitions**:
- → SCIENTIFIC_CHECK (always)

---

### 9. SCIENTIFIC_CHECK Node (ScientificValidatorAgent)

**Purpose**: Validate physics and comparison accuracy

**Checks**:
- [ ] Results are physically reasonable (T+R+A ≈ 1, etc.)
- [ ] No unphysical values (T > 1, negative absorption)
- [ ] Qualitative comparison accurate
- [ ] Quantitative calculations correct
- [ ] Classifications match the data
- [ ] Discrepancies properly documented

**Outputs**:
- `scientific_verdict`: "approve" | "needs_revision"
- `physics_validation`: Conservation laws, value ranges
- `comparison_validation`: Accuracy of paper comparison

**Transitions**:
- → SUPERVISOR (approved)
- → ANALYZE (needs revision, count < 2)
- → SUPERVISOR (needs revision, count >= 2, with flag)

---

### 10. SUPERVISOR Node (SupervisorAgent)

**Purpose**: Big-picture assessment and strategic decisions

**Inputs**:
- `plan`: Full plan
- `assumptions`: All assumptions
- `progress`: Current progress
- `per_figure_reports`: Recent results

**Assessment Criteria**:
1. Is main physics being reproduced?
2. Are validation stages passing?
3. Are we making progress or stuck?
4. Diminishing returns?

**Outputs**:
- `supervisor_verdict`: Decision
- `supervisor_feedback`: Recommendations

**Transitions**:
| Verdict | Next Node |
|---------|-----------|
| ok_continue | SELECT_STAGE |
| change_priority | SELECT_STAGE (reordered) |
| replan_needed | PLAN (if count < 2) |
| replan_needed | ASK_USER (if count >= 2) |
| ask_user | ASK_USER |
| all_complete | GENERATE_REPORT |

---

### 11. ASK_USER Node

**Purpose**: Pause for user input

**Triggers**:
- Revision limits exceeded
- Ambiguous paper information
- Trade-off decisions needed
- Domain expertise required

**Behavior**:
1. Set `awaiting_user_input = True`
2. Populate `pending_user_questions`
3. Pause graph execution
4. Wait for `user_responses` to be filled
5. Resume to appropriate node

---

### 12. GENERATE_REPORT Node (SupervisorAgent)

**Purpose**: Compile final reproduction report

**Triggers**:
- All stages completed
- Reproduction stopped (blocked or user decision)
- `should_stop = true` from Supervisor

**Inputs**:
- All `per_figure_reports` from stages
- `assumptions` log
- `progress` with all discrepancies
- `systematic_shifts` identified

**Outputs**:
- `final_report_markdown`: Complete REPRODUCTION_REPORT_<paper_id>.md
- `overall_assessment`: Executive summary data
- `report_conclusions`: Main findings

**Report Structure**:
1. Executive Summary (Overall Assessment table)
2. Simulation Assumptions (3 tables)
3. Figure-by-Figure Comparisons (side-by-side with tables)
4. Summary Table (all figures at a glance)
5. Systematic Discrepancies (named and explained)
6. Conclusions (key findings, final statement)

---

## Complete Flow Diagram

```
                                    ┌─────────────┐
                                    │    START    │
                                    └──────┬──────┘
                                           │
                                           ▼
                                    ┌─────────────┐
                              ┌─────│    PLAN     │◄────────────────┐
                              │     └──────┬──────┘                 │
                              │            │                        │
                              │            ▼                        │
                              │     ┌─────────────┐                 │
                              │     │SELECT_STAGE │                 │
                              │     └──────┬──────┘                 │
                              │            │                        │
                              │     ┌──────┴──────┐                 │
                              │     │             │                 │
                              │     ▼             ▼                 │
                              │  [has next]   [no more]             │
                              │     │             │                 │
                              │     │             ▼                 │
                              │     │      ┌─────────────┐          │
                              │     │      │GEN_REPORT   │          │
                              │     │      └──────┬──────┘          │
                              │     │             │                 │
                              │     │             ▼                 │
                              │     │      ┌─────────────┐          │
                              │     │      │     END     │          │
                              │     │      └─────────────┘          │
                              │     │                               │
                              │     ▼                               │
                              │  ┌─────────────┐                    │
                     ┌────────┴─►│   DESIGN    │◄─────────┐         │
                     │           │(Sim Designer)│         │         │
                     │           └──────┬──────┘          │         │
                     │                  │                 │         │
                     │                  ▼                 │         │
                     │           ┌─────────────┐          │         │
                     │           │ CODE_REVIEW │          │         │
                     │           │(design check)│         │         │
                     │           └──────┬──────┘          │         │
                     │                  │                 │         │
                     │        ┌─────────┼─────────┐       │         │
                     │        │         │         │       │         │
                     │        ▼         ▼         ▼       │         │
                     │   [approve]  [revise]  [limit]     │         │
                     │        │         │         │       │         │
                     │        │         └─────────┼───────┘         │
                     │        │                   │                 │
                     │        ▼                   │                 │
                     │  ┌─────────────┐           │                 │
                     │  │GENERATE_CODE│◄──────┐   │                 │
                     │  │(CodeGenerator)│      │   │                 │
                     │  └──────┬──────┘       │   │                 │
                     │         │              │   │                 │
                     │         ▼              │   │                 │
                     │  ┌─────────────┐       │   │                 │
                     │  │ CODE_REVIEW │       │   │                 │
                     │  │(code check) │       │   │                 │
                     │  └──────┬──────┘       │   │                 │
                     │         │              │   │                 │
                     │    ┌────┼────┐         │   │                 │
                     │    │    │    │         │   │                 │
                     │    ▼    ▼    ▼         │   │                 │
                     │ [ok] [rev] [limit]     │   │                 │
                     │    │    │    │         │   │                 │
                     │    │    └────┼─────────┘   │                 │
                     │    │         │             │                 │
                     │    ▼         ▼             │                 │
                     │  ┌─────────────┐    ┌─────────────┐          │
                     │  │  RUN_CODE   │    │  ASK_USER   │◄─────┐   │
                     │  └──────┬──────┘    └──────┬──────┘      │   │
                     │         │                  │             │   │
                     │         ▼                  │             │   │
                     │  ┌─────────────┐           │             │   │
                     │  │EXEC_CHECK   │           │             │   │
                     │  │(ExecValidator)│         │             │   │
                     │  └──────┬──────┘           │             │   │
                     │         │                  │             │   │
                     │    ┌────┴────┐             │             │   │
                     │    │         │             │             │   │
                     │    ▼         ▼             │             │   │
                     │  [pass]   [fail]           │             │   │
                     │    │         │             │             │   │
                     │    │         └─────────────┼─────────────┤   │
                     │    │                       │             │   │
                     │    ▼                       │             │   │
                     │  ┌─────────────┐           │             │   │
                     │  │  ANALYZE    │◄──────────┼─────┐       │   │
                     │  │(ResultsAnalyzer)│       │     │       │   │
                     │  └──────┬──────┘           │     │       │   │
                     │         │                  │     │       │   │
                     │         ▼                  │     │       │   │
                     │  ┌─────────────┐           │     │       │   │
                     │  │SCIENCE_CHECK│           │     │       │   │
                     │  │(SciValidator)│          │     │       │   │
                     │  └──────┬──────┘           │     │       │   │
                     │         │                  │     │       │   │
                     │    ┌────┴────┐             │     │       │   │
                     │    │         │             │     │       │   │
                     │    ▼         ▼             │     │       │   │
                     │ [approve] [revise]         │     │       │   │
                     │    │         │             │     │       │   │
                     │    │         └─────────────┼─────┘       │   │
                     │    │                       │             │   │
                     │    ▼                       │             │   │
                     │  ┌─────────────┐           │             │   │
                     │  │ SUPERVISOR  │───────────┴─────────────┘   │
                     │  └──────┬──────┘                             │
                     │         │                                    │
                     │    ┌────┴────────────┐                       │
                     │    │         │       │                       │
                     │    ▼         ▼       ▼                       │
                     │ [continue] [replan] [ask]                    │
                     │    │         │       │                       │
                     └────┘         └───────┴───────────────────────┘
```

## State Persistence

### Checkpointing

The graph supports checkpointing at key points:
- After PLAN completes
- After each stage completes (SUPERVISOR node)
- Before ASK_USER pauses

### File Outputs

State is mirrored to JSON files:
```
outputs/<paper_id>/
├── plan_<paper_id>.json
├── assumptions_<paper_id>.json
├── progress_<paper_id>.json
├── stage0_material_validation/
│   ├── design.json
│   ├── code.py
│   ├── *.csv
│   └── *.png
├── stage1_single_disk/
│   ├── design.json
│   ├── code.py
│   ├── *.csv
│   └── *.png
└── ...
```

## Error Handling

### Simulation Errors

```python
if state["run_error"]:
    # Increment revision count
    state["code_revision_count"] += 1
    
    if state["code_revision_count"] >= MAX_CODE_REVISIONS:
        # Escalate to user
        state["pending_user_questions"].append(
            f"Simulation failed after {MAX_CODE_REVISIONS} attempts. "
            f"Error: {state['run_error']}. How should we proceed?"
        )
        return "ask_user"
    
    # Try again with error context
    state["reviewer_feedback"] = f"Simulation failed: {state['run_error']}"
    return "generate_code"
```

### Validation Hierarchy Enforcement

```python
def check_validation_hierarchy(state):
    """Ensure validation stages pass before proceeding."""
    current_type = state["current_stage_type"]
    
    if current_type == "SINGLE_STRUCTURE":
        if state["validation_hierarchy"]["material_validation"] != "passed":
            return False, "Material validation must pass first"
    
    if current_type in ["ARRAY_SYSTEM", "PARAMETER_SWEEP"]:
        if state["validation_hierarchy"]["single_structure"] not in ["passed", "partial"]:
            return False, "Single structure validation must pass first"
    
    return True, None
```

## Configuration

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4o-mini

# Runtime Limits
MAX_STAGE_RUNTIME_MINUTES=60
TOTAL_RUNTIME_BUDGET_MINUTES=240

# Revision Limits
MAX_DESIGN_REVISIONS=3
MAX_CODE_REVISIONS=3
MAX_ANALYSIS_REVISIONS=2
MAX_REPLANS=2
```

### Model Selection

Different agents can use different models:

```python
from langchain_openai import ChatOpenAI

# Cost-effective for reviews and validation
reviewer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
execution_validator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
scientific_validator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# More capable for design and code generation
designer_llm = ChatOpenAI(model="gpt-4o", temperature=0)
code_generator_llm = ChatOpenAI(model="gpt-4o", temperature=0)
analyzer_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Strategic decisions
supervisor_llm = ChatOpenAI(model="gpt-4o", temperature=0)
```
