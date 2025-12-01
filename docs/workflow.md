# Workflow Documentation

This document describes the LangGraph workflow for paper reproduction.

## Overview

The system uses a state graph where each node represents an agent action or system operation. State flows through the graph, accumulating results and tracking progress.

## Agent Summary (10 Agents)

| Agent | Node | Role |
|-------|------|------|
| PromptAdaptorAgent | ADAPT_PROMPTS | Customizes prompts for paper-specific needs |
| PlannerAgent | PLAN | Reads paper, creates staged plan |
| SimulationDesignerAgent | DESIGN | Designs simulation setup |
| CodeReviewerAgent | CODE_REVIEW | Reviews design and code |
| CodeGeneratorAgent | GENERATE_CODE | Writes Python+Meep code |
| ExecutionValidatorAgent | EXECUTION_CHECK | Validates simulation ran correctly |
| PhysicsSanityAgent | PHYSICS_CHECK | Validates physics (conservation, value ranges) |
| ResultsAnalyzerAgent | ANALYZE | Compares results to paper |
| ComparisonValidatorAgent | COMPARISON_CHECK | Validates comparison accuracy |
| SupervisorAgent | SUPERVISOR | Big-picture decisions |

## Node Definitions

### 1. ADAPT_PROMPTS Node (PromptAdaptorAgent)

**Purpose**: Customize agent prompts for paper-specific requirements

**When**: First node to run, before any other agent

**Inputs**:
- `paper_id`: Unique identifier
- `paper_text`: Extracted paper content (for quick domain scan)

**Outputs**:
- `prompt_modifications`: List of modifications to agent prompts
- `adaptation_log`: Complete log of all changes with reasoning
- `domain_analysis`: Paper domain, materials, techniques identified

**Process**:
1. Quick scan of paper for domain, materials, techniques
2. Identify gaps in base prompts for this paper
3. Generate targeted modifications (append, modify, disable)
4. Document all changes with confidence levels
5. Apply modifications to agent prompts

**Modification Types**:
| Type | Confidence Required | Description |
|------|---------------------|-------------|
| Append | >60% | Add domain-specific guidance |
| Modify | >80% | Adjust existing content |
| Disable | >90% | Mark content as not applicable |

**Constraints**:
- Cannot modify `global_rules.md`
- Cannot change workflow structure
- All changes logged for review

**Transitions**:
- → PLAN (always)

---

### 2. PLAN Node (PlannerAgent)

**Purpose**: Analyze paper and create reproduction plan (using adapted prompt)

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

### 3. SELECT_STAGE Node

**Purpose**: Choose next stage to execute based on dependencies, status, validation hierarchy, and budget

**Logic**:
```python
def select_next_stage(state):
    # Check validation hierarchy first
    hierarchy = state["validation_hierarchy"]
    if not hierarchy["material_validated"]:
        # Cannot proceed past Stage 0 without material validation
        return "stage0_material_validation"
    
    # Check runtime budget
    budget_remaining = state.get("runtime_budget_remaining_seconds", float("inf"))
    
    for stage in state["plan"]["stages"]:
        # Skip completed or blocked stages
        if stage["status"] in ["completed_success", "completed_partial", "blocked"]:
            continue
        
        # Check dependencies
        deps_met = all(
            get_stage_status(dep) in ["completed_success", "completed_partial"]
            for dep in stage["dependencies"]
        )
        
        if not deps_met:
            continue
        
        # Check validation hierarchy requirements
        stage_type = stage.get("stage_type", "")
        if stage_type == "ARRAY_SYSTEM" and not hierarchy["single_structure_validated"]:
            continue  # Cannot do array without single structure
        if stage_type == "PARAMETER_SWEEP" and not hierarchy["array_validated"]:
            continue  # Cannot sweep without array validation
        if stage_type == "COMPLEX_PHYSICS" and not hierarchy["sweep_validated"]:
            continue  # Cannot do complex physics without sweep validation
        
        # Check if budget allows this stage
        estimated_runtime = stage.get("estimated_runtime_seconds", 0)
        if estimated_runtime > budget_remaining:
            # Skip expensive stages when budget is low
            # Mark as blocked with reason
            stage["status"] = "blocked"
            stage["issues"].append(f"Skipped: estimated {estimated_runtime}s exceeds budget {budget_remaining}s")
            continue
        
        return stage["stage_id"]
    
    return None  # All stages done
```

**Transitions**:
- → DESIGN (has next stage)
- → GENERATE_REPORT (no more stages)

---

### 4. DESIGN Node (SimulationDesignerAgent)

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

### 5. CODE_REVIEW Node (CodeReviewerAgent)

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

### 6. GENERATE_CODE Node (CodeGeneratorAgent)

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

### 7. RUN_CODE Node (Python Execution)

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

### 8. EXECUTION_CHECK Node (ExecutionValidatorAgent)

**Purpose**: Validate that simulation ran correctly (technical checks)

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
- `proceed_to_physics`: Boolean

**Transitions**:
- → PHYSICS_CHECK (pass or warning)
- → GENERATE_CODE (fail, recoverable error)
- → ASK_USER (fail, unknown error or limit reached)

---

### 9. PHYSICS_CHECK Node (PhysicsSanityAgent)

**Purpose**: Validate that results are physically reasonable (before comparison)

**Checks**:
- [ ] Conservation laws: T + R + A ≈ 1
- [ ] Value ranges: 0 ≤ T ≤ 1, 0 ≤ R ≤ 1, A ≥ 0
- [ ] No NaN/Inf values
- [ ] Numerical quality (smoothness, symmetry)
- [ ] No boundary artifacts

**Outputs**:
- `physics_verdict`: "pass" | "warning" | "fail"
- `physics_validation`: Conservation, value ranges, numerical quality
- `proceed_to_analysis`: Boolean

**Transitions**:
- → ANALYZE (pass or warning)
- → GENERATE_CODE (fail, suggests code issue)
- → ASK_USER (fail, unknown cause)

---

### 10. ANALYZE Node (ResultsAnalyzerAgent)

**Purpose**: Compare results to paper and classify reproduction quality

**Multimodal Capability**: This node uses vision-capable LLMs (GPT-4o, Claude) to visually compare simulation output images against paper figure images. Both images are provided directly to the LLM.

**Inputs**:
- `stage_outputs`: Files and data from simulation (including generated plot images)
- `plan`: Target figures and validation criteria
- `paper_figures`: Image paths from `PaperInput.figures` for visual comparison

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
- → COMPARISON_CHECK (always)

---

### 11. COMPARISON_CHECK Node (ComparisonValidatorAgent)

**Purpose**: Validate that ResultsAnalyzerAgent's comparison is accurate

**Checks**:
- [ ] All target figures have comparison reports
- [ ] Math is correct (percent differences)
- [ ] Thresholds applied correctly
- [ ] Classifications match the numbers
- [ ] Discrepancies properly documented
- [ ] Progress status consistent with results

**Outputs**:
- `comparison_verdict`: "approve" | "needs_revision"
- `comparison_validation`: Math, classifications, documentation
- `issues`: List of problems found

**Transitions**:
- → SUPERVISOR (approved)
- → ANALYZE (needs revision, count < 2)
- → SUPERVISOR (needs revision, count >= 2, with flag)

---

### 12. SUPERVISOR Node (SupervisorAgent)

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

**MANDATORY CHECKPOINT: Material Validation (Stage 0)**

After Stage 0 completes, the Supervisor MUST trigger a user checkpoint:
- Verdict is ALWAYS `ask_user` after Stage 0 (even if all criteria pass)
- Shows user the material property plots (ε(λ), n(λ), k(λ))
- Asks user to confirm material data sources are correct
- Only proceeds to Stage 1 after user confirmation

This is the highest-leverage checkpoint in the system. If material data is wrong, everything downstream fails.

**Outputs**:
- `supervisor_verdict`: Decision
- `supervisor_feedback`: Recommendations

**Transitions**:
| Verdict | Next Node | Notes |
|---------|-----------|-------|
| ok_continue | SELECT_STAGE | (Cannot use after Stage 0) |
| change_priority | SELECT_STAGE (reordered) | |
| replan_needed | PLAN (if count < 2) | |
| replan_needed | ASK_USER (if count >= 2) | |
| ask_user | ASK_USER | (MANDATORY after Stage 0) |
| all_complete | GENERATE_REPORT | |

---

### 13. ASK_USER Node

**Purpose**: Pause for user input

**Triggers**:
- **Material validation checkpoint** (MANDATORY after Stage 0)
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

### 14. GENERATE_REPORT Node (SupervisorAgent)

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
                                    │ADAPT_PROMPTS│
                                    │(PromptAdapt)│
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
                     │  │(CodeGenerator)│     │   │                 │
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
                     │  │PHYSICS_CHECK│           │             │   │
                     │  │(PhysSanity) │           │             │   │
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
                     │  │COMPARE_CHECK│           │     │       │   │
                     │  │(CompValidator)│         │     │       │   │
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

## State Mutation Contract

This section documents which nodes mutate which state fields, when state is persisted to disk, and checkpoint trigger points.

### Node → State Field Mutations

| Node | Reads | Writes |
|------|-------|--------|
| ADAPT_PROMPTS | paper_text, paper_domain | prompt_adaptations |
| PLAN | paper_text, assumptions | plan, assumptions, extracted_parameters, validation_hierarchy |
| SELECT_STAGE | plan, validation_hierarchy | current_stage_id, current_stage_type |
| DESIGN | plan, assumptions, reviewer_feedback | design_description, performance_estimate, new_assumptions |
| CODE_REVIEW | design_description, code | reviewer_verdict, reviewer_issues, reviewer_feedback |
| GENERATE_CODE | design_description, reviewer_feedback | code |
| RUN_CODE | code | stage_outputs, run_error, runtime_seconds |
| EXECUTION_CHECK | stage_outputs, run_error | execution_valid, execution_issues |
| PHYSICS_SANITY | stage_outputs | physics_valid, physics_issues |
| ANALYZE | stage_outputs, paper_figures | analysis_summary, figure_comparisons, discrepancies_log |
| COMPARISON_CHECK | figure_comparisons, analysis_summary | comparison_valid, comparison_issues |
| SUPERVISOR | analysis_summary, validation_hierarchy | supervisor_verdict, progress, stage status updates |
| ASK_USER | user_question | user_response, awaiting_user_input |
| GENERATE_REPORT | figure_comparisons, progress | report_conclusions, final_report_path |

### State Persistence Rules

1. **In-Memory State**: All fields in `ReproState` are in-memory during execution
2. **Disk Sync Points**:
   - `plan`, `assumptions`, `progress` are synced to JSON files at checkpoints
   - `figure_comparisons` aggregated into report at GENERATE_REPORT
   - `metrics` appended to log file at each agent call
3. **Canonical Sources**:
   - `plan["extracted_parameters"]` is canonical; `state.extracted_parameters` is synced view
   - `DISCREPANCY_THRESHOLDS` in state.py is canonical for threshold values

### Checkpoint Trigger Points

| Checkpoint Name | Trigger Location | Contents Saved |
|-----------------|------------------|----------------|
| `after_plan` | After PLAN node | Full state with plan, assumptions |
| `after_stage0_user_confirm` | After user confirms materials | State + user confirmation |
| `after_stage_N_complete` | After each SUPERVISOR approval | Full state with stage progress |
| `before_ask_user` | Before ASK_USER node | Full state for resume |
| `final_report` | After GENERATE_REPORT | Full state + report path |

## State Persistence

### Checkpointing

The graph supports checkpointing at key points:
- `after_plan`: After PlannerAgent completes the reproduction plan
- `after_stage_complete`: After each stage completes (SUPERVISOR node)
- `before_ask_user`: Before pausing for user input

### Checkpoint Functions

```python
from schemas.state import save_checkpoint, load_checkpoint, list_checkpoints

# Save checkpoint after planning
checkpoint_path = save_checkpoint(state, "after_plan")
# Creates: outputs/<paper_id>/checkpoints/checkpoint_<paper_id>_after_plan_<timestamp>.json

# Save checkpoint after stage completion
checkpoint_path = save_checkpoint(state, f"stage{stage_num}_complete")

# List all checkpoints for a paper
checkpoints = list_checkpoints("aluminum_nanoantenna_2013")
# Returns: [{"name": "after_plan", "timestamp": "20251130_143022", "path": "...", "size_kb": 45.2}, ...]

# Resume from checkpoint
state = load_checkpoint("aluminum_nanoantenna_2013", checkpoint_name="after_plan")
# Or load most recent:
state = load_checkpoint("aluminum_nanoantenna_2013", checkpoint_name="latest")
```

### Resume Behavior

When resuming from a checkpoint:
1. Load state from checkpoint file
2. Identify current workflow phase from `workflow_phase` field
3. Resume execution at the appropriate node:
   - `"planning"` → Resume at PLAN node
   - `"design"` → Resume at DESIGN node
   - `"running"` → Resume at RUN_CODE node
   - `"analysis"` → Resume at ANALYZE node
   - etc.

### Checkpoint File Structure

```
outputs/<paper_id>/checkpoints/
├── checkpoint_<paper_id>_after_plan_20251130_143022.json
├── checkpoint_after_plan_latest.json  (symlink to most recent)
├── checkpoint_<paper_id>_stage1_complete_20251130_144515.json
├── checkpoint_stage1_complete_latest.json
└── ...
```

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

# Cost-effective for validation (focused checks)
execution_validator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
physics_sanity_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
comparison_validator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
reviewer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# More capable for design and analysis
designer_llm = ChatOpenAI(model="gpt-4o", temperature=0)
code_generator_llm = ChatOpenAI(model="gpt-4o", temperature=0)
analyzer_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Strategic decisions
supervisor_llm = ChatOpenAI(model="gpt-4o", temperature=0)
```

## Context Management

### How Context Works in LangGraph

Each agent call is an **independent LLM invocation** with its own context window. Understanding this is critical for system design:

**Between Different Agents:**
- State is shared via the `ReproState` TypedDict (passed between nodes)
- Full conversation history is NOT shared between agents
- Each agent receives only its system prompt + current state + node-specific inputs

**Between Steps of the Same Agent (Loops):**
- If an agent loops (e.g., CodeReviewerAgent says "revise" → back to CodeGeneratorAgent → back to CodeReviewerAgent)
- The second call gets **fresh context** with updated state
- Previous conversation turns are NOT carried over
- Feedback is passed via state fields (e.g., `reviewer_feedback`, `critic_issues`)

### What Each Agent Receives

```
┌─────────────────────────────────────────────┐
│              Agent Context                  │
├─────────────────────────────────────────────┤
│ 1. System Prompt (from prompts/*.md)        │
│    - Agent role and responsibilities        │
│    - Checklists and guidelines              │
│    - Output format requirements             │
│                                             │
│ 2. Global Rules (prepended)                 │
│    - Non-negotiable rules from global_rules │
│                                             │
│ 3. Current State (from ReproState)          │
│    - paper_id, paper_text, paper_domain     │
│    - plan, assumptions, progress            │
│    - current_stage_id, workflow_phase       │
│    - Revision counts, verdicts              │
│                                             │
│ 4. Node-Specific Inputs                     │
│    - reviewer_feedback (if revising)        │
│    - stage_outputs (after run)              │
│    - paper_figures (for comparison)         │
└─────────────────────────────────────────────┘
```

### Why This Design

**Benefits:**
- Prevents context bloat (agents don't see irrelevant history)
- Each agent focuses on its specific task
- State provides structured, relevant information
- Easier to debug (each call is self-contained)

**Implications:**
- Important information must be explicitly saved to state
- Feedback between agents must use state fields
- Large outputs should be summarized before storing

### Feedback Loop Example

```
CodeGeneratorAgent (attempt 1)
    ↓ generates code
CodeReviewerAgent
    ↓ finds issues, sets reviewer_feedback="Missing progress prints"
    ↓ sets last_critic_verdict="needs_revision"
CodeGeneratorAgent (attempt 2)
    ↓ receives: fresh prompt + state including reviewer_feedback
    ↓ uses feedback to improve code
CodeReviewerAgent
    ↓ reviews improved code
    ↓ sets last_critic_verdict="approve_to_run"
```
