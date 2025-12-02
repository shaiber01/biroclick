# Workflow Documentation

> **Design Document**: Code examples below are illustrative specifications. 
> See `src/` for actual implementation status.

This document describes the LangGraph workflow for paper reproduction.

## Overview

The system uses a state graph where each node represents an agent action or system operation. State flows through the graph, accumulating results and tracking progress.

## Agent Summary (12 Agents)

| Agent | Node | Role |
|-------|------|------|
| PromptAdaptorAgent | adapt_prompts | Customizes prompts for paper-specific needs |
| PlannerAgent | plan | Reads paper, creates staged plan |
| PlanReviewerAgent | plan_review | Reviews reproduction plan before stage execution |
| SimulationDesignerAgent | design | Designs simulation setup |
| DesignReviewerAgent | design_review | Reviews simulation design before code generation |
| CodeGeneratorAgent | generate_code | Writes Python+Meep code |
| CodeReviewerAgent | code_review | Reviews generated code before execution |
| ExecutionValidatorAgent | execution_check | Validates simulation ran correctly |
| PhysicsSanityAgent | physics_check | Validates physics (conservation, value ranges) |
| ResultsAnalyzerAgent | analyze | Compares results to paper |
| ComparisonValidatorAgent | comparison_check | Validates comparison accuracy |
| SupervisorAgent | supervisor | Big-picture decisions |

## Data Ownership Matrix

Clear ownership of state artifacts prevents conflicts and ensures single source of truth:

| Artifact | Created By | Updated By | Read By |
|----------|------------|------------|---------|
| `plan` | PlannerAgent | Never (immutable after approval) | All agents |
| `plan["stages"]` | PlannerAgent | Never | Use `get_plan_stage()` to look up design specs |
| `progress` | select_stage_node (init) | All validators, supervisor | SupervisorAgent, Report |
| `progress["stages"]` | initialize_progress_from_plan() | All stage executors | Reference plan by stage_id for design specs |
| `assumptions` | PlannerAgent (initial) | SupervisorAgent (user corrections) | All agents |
| `validated_materials` | material_checkpoint_node | Never (immutable) | CodeGeneratorAgent |
| `code` | CodeGeneratorAgent | CodeGeneratorAgent (revisions) | CodeReviewerAgent, ExecutionValidator |
| `design_description` | SimulationDesignerAgent | SimulationDesignerAgent (revisions) | DesignReviewerAgent, CodeGenerator |
| `stage_outputs` | code_runner | Never | ResultsAnalyzer, PhysicsSanity |
| `extracted_parameters` | PlannerAgent | sync_extracted_parameters() | All agents |
| `validation_hierarchy` | Computed via get_validation_hierarchy() | N/A (derived) | SupervisorAgent |

**Ownership Rules:**
- **"Created By"** agent is the authoritative source for that data
- **"Updated By"** agents can modify only via defined mechanisms (e.g., user corrections via Supervisor)
- **"Never"** means immutable after initial creation - modifications require new creation
- **Computed** fields are derived on-demand and never stored directly

### Plan vs Progress Separation

**Critical Design Principle**: Plan stages and progress stages serve different purposes:

| Aspect | `plan["stages"]` | `progress["stages"]` |
|--------|------------------|---------------------|
| **Contains** | Design specs (WHAT should happen) | Execution state (WHAT did happen) |
| **Mutability** | Immutable after approval | Mutable during execution |
| **Fields** | expected_outputs, validation_criteria, runtime_budget_minutes, complexity_class | status, outputs, discrepancies, runtime_seconds, issues |
| **Access** | `get_plan_stage(state, stage_id)` | `get_progress_stage(state, stage_id)` |

**Why This Matters:**
- **No duplication**: Design specs exist only in plan, not copied to progress
- **No drift risk**: Can't have progress.expected_outputs diverge from plan.expected_outputs
- **Clear semantics**: Plan = contract, Progress = execution log

**Usage Pattern:**
```python
# WRONG: Looking for expected_outputs in progress (won't exist)
progress_stage = get_progress_stage(state, "stage1")
outputs = progress_stage.get("expected_outputs", [])  # ❌ Empty!

# CORRECT: Get design specs from plan, execution state from progress
plan_stage = get_plan_stage(state, "stage1")
expected = plan_stage.get("expected_outputs", [])  # ✅ Design spec

progress_stage = get_progress_stage(state, "stage1")
actual = progress_stage.get("outputs", [])  # ✅ What was produced
```

## Agent Call Pattern

All agent nodes follow this standardized pattern to ensure consistency and schema compliance:

### 1. Input Assembly

Each agent reads only the state fields it needs (defined in `NODE_REQUIREMENTS` in `state.py`):

```python
def agent_node(state: ReproState) -> dict:
    # Read only the fields this agent needs
    context = {
        "paper_id": state["paper_id"],
        "plan": state.get("plan", {}),
        "current_stage_id": state.get("current_stage_id"),
        # ... agent-specific fields from NODE_REQUIREMENTS
    }
```

### 2. LLM Call with Schema-Based Function Calling

All agents use structured output via JSON schemas (no free-form text parsing):

```python
    # Load agent's output schema from schemas/*_output_schema.json
    schema = load_output_schema("agent_name")
    
    # Build prompt from template + context
    prompt = build_prompt("agent_name", context)
    
    # Call LLM with function calling (schema enforced)
    response = llm.invoke(prompt, tools=[schema])
```

### 3. Validate and Merge

Return only the fields this agent is allowed to update (per Ownership Matrix):

```python
    # Response is already validated against schema by LLM
    output = response.tool_calls[0].args
    
    # Return only the fields this agent is allowed to update
    return {
        "field_to_update": output["field"],
        "another_field": output["other"],
    }
```

### Output Schema Files

Each agent has a corresponding schema in `schemas/`:

| Agent | Schema File |
|-------|-------------|
| SupervisorAgent | `supervisor_output_schema.json` |
| PlanReviewerAgent | `plan_reviewer_output_schema.json` |
| DesignReviewerAgent | `design_reviewer_output_schema.json` |
| CodeReviewerAgent | `code_reviewer_output_schema.json` |
| ResultsAnalyzerAgent | `results_analyzer_output_schema.json` |
| ExecutionValidatorAgent | `execution_validator_output_schema.json` |
| PhysicsSanityAgent | `physics_sanity_output_schema.json` |
| ComparisonValidatorAgent | `comparison_validator_output_schema.json` |

### Benefits of This Pattern

1. **No parsing failures**: LLM outputs are validated against schema at call time
2. **Clear contracts**: Each agent has explicit input fields and output schema
3. **Easy debugging**: Schema violations surface immediately
4. **Consistent structure**: All agents follow the same pattern

## Node Definitions

### 1. adapt_prompts Node (PromptAdaptorAgent)

**Purpose**: Customize agent prompts for paper-specific requirements

**When**: First node to run, before any other agent

**Inputs**:
- `paper_id`: Unique identifier
- `paper_text_summary`: Abstract + Methods section (critical for material sources, geometry details)

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
- → plan (always)

---

### 2. plan Node (PlannerAgent)

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
- → plan_review (always)

---

### 3. select_stage Node

**Purpose**: Choose the next stage based on the mutable execution log (`progress["stages"]`), respecting the immutable plan contract, validation hierarchy, runtime budget, and any backtracking state.

**Logic**:
```python
def select_next_stage(state):
    progress = state.get("progress", {})
    progress_stages = progress.get("stages", [])
    plan = state.get("plan", {})
    plan_stages = {s["stage_id"]: s for s in plan.get("stages", [])}

    if not progress_stages:
        return None  # Progress hasn't been initialized yet

    # Helper reads status from progress only (plan never mutated after approval)
    def get_progress_status(stage_id: str) -> str:
        stage = next((s for s in progress_stages if s.get("stage_id") == stage_id), None)
        return stage.get("status", "not_started") if stage else "not_started"

    # ─── PRIORITY 1: Handle backtracked stages ───────────────────────────
    # handle_backtrack marks the target stage's progress status as "needs_rerun".
    for stage in progress_stages:
        if stage.get("status") == "needs_rerun":
            # Safety check: dependencies should never be invalidated once we're rerunning
            for dep in stage.get("dependencies", []):
                if get_progress_status(dep) == "invalidated":
                    raise ValueError(
                        f"BUG: needs_rerun stage {stage['stage_id']} has "
                        f"invalidated dependency {dep}. Progress bookkeeping drifted from plan."
                    )
            return stage["stage_id"]

    # ─── PRIORITY 2: Promote invalidated stages once deps recover ────────
    for stage in progress_stages:
        if stage.get("status") == "invalidated":
            deps_ok = all(
                get_progress_status(dep) in ["completed_success", "completed_partial"]
                for dep in stage.get("dependencies", [])
            )
            if deps_ok:
                stage["status"] = "needs_rerun"  # update progress entry, not the plan
                return stage["stage_id"]

    # ─── PRIORITY 3: Normal stage selection ──────────────────────────────
    hierarchy = get_validation_hierarchy(state)  # Derived from progress["stages"]
    if hierarchy["material_validation"] not in ["passed", "partial"]:
        return "stage0_material_validation"  # Stage 0 gate

    budget_remaining = state.get("runtime_budget_remaining_seconds", float("inf"))
    plan_stage_types = {s.get("stage_type") for s in plan.get("stages", [])}

    for progress_stage in progress_stages:
        status = progress_stage.get("status", "not_started")
        if status in ["completed_success", "completed_partial", "blocked", "invalidated", "needs_rerun"]:
            continue

        deps_met = all(
            get_progress_status(dep) in ["completed_success", "completed_partial"]
            for dep in progress_stage.get("dependencies", [])
        )
        if not deps_met:
            continue

        plan_stage = plan_stages.get(progress_stage["stage_id"], {})
        stage_type = plan_stage.get("stage_type", "")

        # Enforce adaptive validation hierarchy using plan metadata
        if stage_type == "ARRAY_SYSTEM":
            if "SINGLE_STRUCTURE" in plan_stage_types and hierarchy["single_structure"] not in ["passed", "partial"]:
                continue
        elif stage_type == "PARAMETER_SWEEP":
            requires_single = "SINGLE_STRUCTURE" in plan_stage_types
            requires_array = "ARRAY_SYSTEM" in plan_stage_types
            if requires_array and hierarchy["arrays_systems"] not in ["passed", "partial"]:
                continue
            if requires_single and not requires_array and hierarchy["single_structure"] not in ["passed", "partial"]:
                continue
        elif stage_type == "COMPLEX_PHYSICS":
            if "PARAMETER_SWEEP" in plan_stage_types and hierarchy["parameter_sweeps"] not in ["passed", "partial"]:
                continue
            if "ARRAY_SYSTEM" in plan_stage_types and hierarchy["arrays_systems"] not in ["passed", "partial"]:
                continue
            if "SINGLE_STRUCTURE" in plan_stage_types and hierarchy["single_structure"] not in ["passed", "partial"]:
                continue

        estimated_runtime = plan_stage.get("estimated_runtime_seconds", 0)
        if estimated_runtime > budget_remaining:
            progress_stage["status"] = "blocked"
            issues = progress_stage.setdefault("issues", [])
            issues.append(
                f"Skipped: estimated {estimated_runtime}s exceeds remaining budget {budget_remaining}s"
            )
            continue

        return progress_stage["stage_id"]

    return None  # All stages done
```

**Stage Status Values**:
| Status | Meaning |
|--------|---------|
| not_started | Stage hasn't been attempted |
| in_progress | Stage is currently executing |
| completed_success | Stage completed with good results |
| completed_partial | Stage completed with partial match |
| completed_failed | Stage completed but failed validation |
| blocked | Stage skipped due to budget/dependencies |
| needs_rerun | Stage needs to be re-executed (backtrack target) |
| invalidated | Stage results invalid, will re-run when deps ready |

**Transitions**:
- → design (has next stage)
- → generate_report (no more stages)

---

### 4. design Node (SimulationDesignerAgent)

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
- → design_review (always)

---

### 5. Review Nodes (Plan / Design / Code)

ReproLab now relies on three dedicated review agents—each with its own prompt, schema, counters, and routing logic. There is no shared `review_context` flag anymore; ownership is split per artifact to keep feedback scoped and auditable.

#### 5.1 plan_review Node (PlanReviewerAgent)

**Purpose**: Validate the staged reproduction plan before any execution.

**Checklist highlights**:
- Every reproducible figure is staged with validation criteria and expected outputs.
- Stage 0 (materials) exists and precedes downstream stages.
- Dependencies form an acyclic graph that respects the adaptive validation hierarchy.
- Targets marked `precision_requirement="excellent"` provide a `digitized_data_path`.
- Runtime budgets align with `runtime_config` limits.

**Outputs**:
- `last_plan_review_verdict` ("approve" | "needs_revision")
- Structured findings defined in `plan_reviewer_output_schema.json`
- `planner_feedback` plus `replan_count` bookkeeping when revisions occur

**Transitions**:
- `approve` → `select_stage`
- `needs_revision` (replan_count < limit) → `plan`
- `needs_revision` (limit reached) → `ask_user`

#### 5.2 design_review Node (DesignReviewerAgent)

**Purpose**: Gate the SimulationDesigner’s artifact before code generation.

**Checklist highlights**:
- Geometry, resolution, and units match plan specs.
- Stage 1+ uses `validated_materials`; Stage 0 uses `planned_materials`.
- Sources, boundary conditions, and monitors map to the physics goal.
- New assumptions are explicitly recorded and scoped.
- Runtime estimates respect the stage’s complexity class.

**Outputs**:
- `last_design_review_verdict`
- `reviewer_issues` + narrative feedback
- `design_revision_count` increments when verdict is `needs_revision`

**Transitions**:
- `approve` → `generate_code`
- `needs_revision` (count < limit) → `design`
- `needs_revision` (limit reached) → `ask_user`

#### 5.3 code_review Node (CodeReviewerAgent)

**Purpose**: Ensure generated Meep code faithfully implements the approved design before sandbox execution.

**Checklist highlights**:
- Implements geometry/material/unit system exactly as reviewed.
- Pulls material files from `validated_materials`; no hard-coded constants.
- Produces expected outputs (names + directories) for downstream analyzers.
- Includes deterministic logging and avoids blocking calls (e.g., `plt.show`).
- Respects sandbox rules (no arbitrary filesystem or network access).

**Outputs**:
- `last_code_review_verdict`
- `reviewer_issues` + actionable feedback
- `code_revision_count` increments on `needs_revision`

**Transitions**:
- `approve` → `run_code`
- `needs_revision` (count < limit) → `generate_code`
- `needs_revision` (limit reached) → `ask_user`

---

### 6. generate_code Node (CodeGeneratorAgent)

**Purpose**: Generate Python+Meep code from approved design

**Inputs**:
- `design`: Approved simulation design from SimulationDesignerAgent
- `reviewer_feedback`: Feedback if revising code

**Outputs**:
- `code`: Complete Python+Meep simulation code
- `expected_outputs`: List of expected output files
- `estimated_runtime_minutes`: Runtime estimate

**Transitions**:
- → code_review (always - code review phase)

---

### 7. run_code Node (Python Execution)

**Purpose**: Execute the simulation code in a sandboxed subprocess

**Implementation**: See `src/code_runner.py` for full implementation.

```python
from src.code_runner import run_code_node

# The run_code_node function:
# 1. Validates code for dangerous/blocking patterns
# 2. Executes in subprocess with timeout and memory limits
# 3. Captures stdout, stderr, and output files
# 4. Returns structured result with error handling

def get_current_stage(state):
    """Helper to get the current stage config from state."""
    current_id = state["current_stage_id"]
    for stage in state["plan"]["stages"]:
        if stage["stage_id"] == current_id:
            return stage
    raise ValueError(f"Stage {current_id} not found in plan")

def run_code_node(state):
    """
    LangGraph node for run_code.
    
    Sandboxing features:
    - Subprocess isolation
    - Configurable timeout (from stage runtime_budget_minutes)
    - Memory limits (from runtime_config.max_memory_gb)
    - Thread/core limits via environment variables
    - Code validation before execution
    """
    from src.code_runner import run_simulation, validate_code
    
    # Validate code first
    warnings = validate_code(state["code"])
    blocking = [w for w in warnings if w.startswith("BLOCKING")]
    if blocking:
        return {"run_error": f"Code validation failed: {blocking}"}
    
    # Get current stage config
    current_stage = get_current_stage(state)
    
    # Execute with sandboxing
    result = run_simulation(
        code=state["code"],
        stage_id=state["current_stage_id"],
        output_dir=Path(f"outputs/{state['paper_id']}/{state['current_stage_id']}"),
        config={
            "timeout_seconds": current_stage["runtime_budget_minutes"] * 60,
            "max_memory_gb": state.get("runtime_config", {}).get("max_memory_gb", 8.0),
            "max_cpu_cores": state.get("runtime_config", {}).get("max_cpu_cores", 4),
        }
    )
    
    return {
        "stage_outputs": {
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "files": result["output_files"],
            "exit_code": result["exit_code"],
            "runtime_seconds": result["runtime_seconds"],
        },
        "run_error": result["error"]
    }
```

**Sandboxing limitations (v1)**:
- No network isolation (simulations shouldn't need network)
- Filesystem limited to working directory
- Memory limits are soft limits on Unix
- Windows requires different approach

See `docs/guidelines.md` Section 14 for sandboxing details and future Docker support.

**Transitions**:
- → EXECUTION_CHECK (always)

---

### 8. EXECUTION_CHECK Node (ExecutionValidatorAgent)

**Purpose**: Validate that simulation ran correctly (technical checks)

**Architectural Note**: This agent is the **SINGLE POINT for failure interpretation**.

- `run_code` returns raw results (`stage_outputs`, `run_error`) without interpretation
- `ExecutionValidatorAgent` is the only agent that decides:
  - Whether to retry code generation (recoverable error)
  - Whether to escalate to user (unknown error)
  - Whether to proceed to physics validation
- This keeps failure handling logic centralized, not scattered across nodes

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
- → physics_check (pass or warning)
- → generate_code (fail, recoverable error)
- → ask_user (fail, unknown error or limit reached)

---

### 9. physics_check Node (PhysicsSanityAgent)

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
- → analyze (pass or warning)
- → generate_code (fail, suggests code issue)
- → ask_user (fail, unknown cause)

---

### 10. analyze Node (ResultsAnalyzerAgent)

**Purpose**: Compare results to paper and classify reproduction quality

**Multimodal Capability**: This node uses Claude Opus 4.5 (vision-capable) to visually compare simulation output images against paper figure images. Both images are provided directly to the LLM.

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
- → supervisor (approved)
- → analyze (needs revision, count < 2)
- → supervisor (needs revision, count >= 2, with flag)

---

### 12. supervisor Node (SupervisorAgent)

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

**Material Validation - User Response Handling:**

| User Response | System Action | Counts Against Limits? |
|--------------|---------------|------------------------|
| "Materials look correct" | Proceed to Stage 1 | No |
| "Wrong database - use Johnson-Christy instead of Palik" | Invalidate Stage 0, update assumptions, re-run | Yes (backtrack count +1) |
| "Wrong material - should be gold not silver" | Invalidate Stage 0, update plan, re-run | Yes (backtrack count +1) |
| "Not sure / need help comparing" | Show comparison of available databases, ask again | No (clarification only) |

**Re-run behavior (database/material change):**
```python
if user_response.verdict == "change_database":
    # Update assumptions with new database choice
    state["assumptions"]["material_sources"][material] = user_response.database_choice
    
    # Invalidate Stage 0 for re-run
    state["progress"]["stages"]["stage0"]["status"] = "needs_rerun"
    
    # Increment backtrack count (limit: 2 total)
    state["backtrack_count"] += 1
    
    # Log the user decision
    log_user_interaction(state, "material_checkpoint", user_response)
    
    # SimulationDesignerAgent will use updated assumptions on re-run

elif user_response.verdict == "change_material":
    # This is more severe - wrong material identified
    # Need to update plan, not just assumptions
    state["supervisor_feedback"] = f"User identified wrong material: use {user_response.material_choice}"
    state["progress"]["stages"]["stage0"]["status"] = "needs_rerun"
    state["backtrack_count"] += 1
    
    # Route to plan to update material selection
    return "plan"

elif user_response.verdict == "need_help":
    # Show comparison and ask again (no penalty)
    state["pending_user_questions"].append(
        format_material_comparison_table(available_databases)
    )
    return "ask_user"  # Same question with more context
```

**Outputs**:
- `supervisor_verdict`: Decision
- `supervisor_feedback`: Recommendations

**Transitions**:
| Verdict | Next Node | Notes |
|---------|-----------|-------|
| ok_continue | select_stage | (Cannot use after Stage 0) |
| change_priority | select_stage (reordered) | |
| replan_needed | plan (if count < 2) | |
| replan_needed | ask_user (if count >= 2) | |
| backtrack_to_stage | handle_backtrack | Cross-stage backtracking |
| ask_user | ask_user | (MANDATORY after Stage 0) |
| all_complete | generate_report | |

**Validation Hierarchy (Computed On-Demand)**

The validation hierarchy is **computed from progress["stages"]** on demand using
`get_validation_hierarchy(state)`. There is NO separate hierarchy state to sync.

**How It Works:**

1. Stage status is stored in `progress["stages"][idx]["status"]`
2. When you need the validation hierarchy, call `get_validation_hierarchy(state)`
3. The function computes hierarchy values from stage statuses automatically

```python
from schemas.state import get_validation_hierarchy

# In select_stage or any agent that needs hierarchy:
hierarchy = get_validation_hierarchy(state)

if hierarchy["material_validation"] != "passed":
    # Cannot proceed past Stage 0
    pass

if hierarchy["single_structure"] in ["passed", "partial"]:
    # Can proceed to array/sweep stages
    pass
```

**Terminology Mapping** (handled automatically by `get_validation_hierarchy()`):

| Stage Status | Hierarchy Value |
|--------------|-----------------|
| `completed_success` | `passed` |
| `completed_partial` | `partial` |
| `completed_failed` | `failed` |
| `not_started`, `in_progress`, `blocked`, `needs_rerun`, `invalidated` | `not_done` |

**Why This Design:**

- **Single source of truth**: Stage status in `progress["stages"]` is the only data
- **No sync bugs**: Hierarchy is derived, not stored, so can't get out of sync
- **Simple updates**: Just update stage status; hierarchy reflects it automatically

**What Agents Do:**

1. Update `progress["stages"][idx]["status"]` when stage completes
2. Call `get_validation_hierarchy(state)` when checking if stages can run
3. That's it - no manual sync needed

---

### 13. material_checkpoint Node

**Purpose**: Mandatory user confirmation after Stage 0 that populates `validated_materials` for all subsequent stages.

**When Called**: After SupervisorAgent sets `ask_user` verdict following Stage 0 completion.

**Why This Node Exists**:
This is a **critical handoff point** in the workflow. The `validated_materials` field is:
- **Required** by CodeGeneratorAgent for Stage 1+ (see `NODE_REQUIREMENTS`)
- **Never modified** after creation (immutable per Data Ownership Matrix)
- **Only populated here** after user confirmation

Without this node, Stage 1+ would have no material paths and would fail.

**Inputs**:
- `planned_materials`: Materials identified by PlannerAgent from paper analysis
- `stage_outputs`: Stage 0 output files (material property plots, ε/n/k data)
- `user_responses`: User confirmation or correction

**Logic**:
```python
def material_checkpoint_node(state):
    """
    Populate validated_materials after user confirms Stage 0 results.
    
    This is the ONLY place validated_materials gets populated.
    CodeGeneratorAgent for Stage 1+ reads from validated_materials, NOT planned_materials.
    """
    user_response = state.get("user_responses", {})
    
    # Check what the user decided
    material_verdict = user_response.get("material_validation", {})
    verdict = material_verdict.get("verdict", "")
    
    if verdict == "approve":
        # ═══════════════════════════════════════════════════════════════════
        # SUCCESS PATH: Populate validated_materials from planned_materials
        # ═══════════════════════════════════════════════════════════════════
        validated = []
        for mat in state.get("planned_materials", []):
            validated.append({
                "material_id": mat["material_id"],
                "name": mat["name"],
                "source": mat.get("source", "unknown"),
                "path": mat["path"],  # Path to material data file
                "validated_by_user": True,
                "validation_timestamp": datetime.now().isoformat(),
            })
        
        # Log the user interaction
        log_user_interaction(
            state,
            interaction_type="material_checkpoint",
            question=state.get("pending_user_questions", [""])[0],
            response=material_verdict,
            context={"stage_id": "stage0_material_validation", "agent": "material_checkpoint"}
        )
        
        # Clear pending questions
        state["pending_user_questions"] = []
        state["awaiting_user_input"] = False
        
        return {
            "validated_materials": validated,
            "pending_user_questions": [],
            "awaiting_user_input": False,
            "user_responses": {},  # Clear for next interaction
        }
    
    elif verdict == "change_database":
        # ═══════════════════════════════════════════════════════════════════
        # USER WANTS DIFFERENT DATABASE: Update assumptions, re-run Stage 0
        # ═══════════════════════════════════════════════════════════════════
        new_database = material_verdict.get("database_choice")
        material_id = material_verdict.get("material_id")
        
        # Update planned_materials with new database
        updated_materials = []
        for mat in state.get("planned_materials", []):
            if mat["material_id"] == material_id:
                # Replace with new database path
                mat = {**mat, "source": new_database, "path": f"materials/{new_database}_{mat['name'].lower()}.csv"}
            updated_materials.append(mat)
        
        # Mark Stage 0 for re-run
        update_progress_stage_status(state, "stage0_material_validation", "needs_rerun",
            summary=f"User requested material database change: {new_database}")
        
        log_user_interaction(
            state,
            interaction_type="material_checkpoint",
            question=state.get("pending_user_questions", [""])[0],
            response=material_verdict,
            context={"stage_id": "stage0_material_validation", "action": "database_change"}
        )
        
        return {
            "planned_materials": updated_materials,
            "backtrack_count": state.get("backtrack_count", 0) + 1,
            "pending_user_questions": [],
            "awaiting_user_input": False,
            "user_responses": {},
        }
    
    elif verdict == "change_material":
        # ═══════════════════════════════════════════════════════════════════
        # WRONG MATERIAL: This requires replanning (more severe)
        # ═══════════════════════════════════════════════════════════════════
        log_user_interaction(
            state,
            interaction_type="material_checkpoint",
            question=state.get("pending_user_questions", [""])[0],
            response=material_verdict,
            context={"stage_id": "stage0_material_validation", "action": "material_change"}
        )
        
        return {
            "planner_feedback": f"User identified wrong material: use {material_verdict.get('material_choice')}",
            "replan_count": state.get("replan_count", 0) + 1,
            "backtrack_count": state.get("backtrack_count", 0) + 1,
            "pending_user_questions": [],
            "awaiting_user_input": False,
            "user_responses": {},
        }
    
    else:
        # User needs more information - keep awaiting input
        return {"awaiting_user_input": True}
```

**Outputs**:
| User Verdict | Output Fields | Next Node |
|--------------|---------------|-----------|
| `approve` | `pending_validated_materials` populated (awaiting supervisor move) | ask_user |
| `change_database` | `planned_materials` updated, Stage 0 `needs_rerun` | ask_user |
| `change_material` | `planner_feedback` set | ask_user |
| `need_help` | Additional context added | ask_user |

**Transitions**:
- → ask_user (always; mandatory pause for user confirmation)
- Supervisor then routes to `select_stage`, `plan`, etc. based on the recorded response.

**STATE INVARIANT**: `material_checkpoint` never mutates `validated_materials` directly. Materials remain in `pending_validated_materials` until Supervisor processes the user’s approval and moves them, guaranteeing the human-in-the-loop gate.

---

### 14. handle_backtrack Node

**Purpose**: Process cross-stage backtracking when Supervisor accepts a backtrack suggestion

**Triggers**:
- `supervisor_verdict = "backtrack_to_stage"`
- Supervisor accepted a backtrack suggestion from another agent

**Inputs**:
- `backtrack_decision` from SupervisorAgent containing:
  - `target_stage_id`: Stage to go back to
  - `stages_to_invalidate`: List of stages to mark as needing re-run
  - `reason`: Why backtracking is needed

**Logic**:
```python
def handle_backtrack(state):
    """
    Process backtrack decision. Updates stage statuses directly.
    
    The validation hierarchy is computed on-demand from stage statuses,
    so no manual hierarchy sync is needed. Just update the status fields
    and get_validation_hierarchy() will reflect the changes automatically.
    """
    decision = state["backtrack_decision"]
    target_stage = decision["target_stage_id"]
    progress = state["progress"]
    
    # Find and update stage statuses
    for stage in progress.get("stages", []):
        stage_id = stage.get("stage_id")
        if stage_id == target_stage:
            # Target stage: mark for re-run
            stage["status"] = "needs_rerun"
            # Note: get_validation_hierarchy() will now return "not_done" for this stage
        elif stage_id in decision["stages_to_invalidate"]:
            # Dependent stages: mark as invalidated
            stage["status"] = "invalidated"
            stage["invalidation_reason"] = decision["reason"]
            # Note: get_validation_hierarchy() will now return "not_done" for these stages
    
    # Increment backtrack counter
    state["backtrack_count"] += 1
    
    # Clear working data to start fresh
    state["current_stage_id"] = None
    state["code"] = None
    state["design_description"] = None
    state["backtrack_suggestion"] = None  # Clear the suggestion
    
    # Log the backtrack event
    log_backtrack_event(state, decision)
    
    return state
```

**Outputs**:
- Updated `progress` with invalidated stages
- Incremented `backtrack_count`
- Cleared working data

**Transitions**:
- → select_stage (always)

#### Backtracking Semantics Reference

This table summarizes when each agent can propose backtracking and what typically gets invalidated.

**Which Agents Can Propose Backtracking?**

| Agent | Can Propose? | In Output Field | Authority Level |
|-------|-------------|-----------------|-----------------|
| **ResultsAnalyzerAgent** | ✅ Yes | `backtrack_suggestion` | Suggests only |
| **PhysicsSanityAgent** | ✅ Yes | `backtrack_suggestion` | Suggests only |
| **CodeReviewerAgent** | ✅ Yes | `backtrack_suggestion` | Suggests only |
| **SupervisorAgent** | ✅ Yes (decides) | `backtrack_decision` | **Final authority** |
| All other agents | ❌ No | N/A | N/A |

**When to Propose Backtracking (by Agent)**

| Agent | Propose Backtrack When... | Do NOT Backtrack For... |
|-------|---------------------------|------------------------|
| **ResultsAnalyzerAgent** | Wrong geometry type discovered (spheres vs rods), wrong material identified, wrong wavelength range | Minor quantitative discrepancies, expected approximation errors |
| **PhysicsSanityAgent** | Physics evidence of fundamental setup error (wrong mode type, impossible results) | Numerical noise, minor conservation law deviations |
| **CodeReviewerAgent** | Design was based on wrong assumptions from earlier stages | Code bugs fixable in current stage, style issues |

**Typical `invalidated_stages` by Error Type**

| Error Discovered | Backtrack To | Invalidated Stages | Example |
|-----------------|--------------|-------------------|---------|
| Wrong material data | Stage 0 | All stages 1+ | Used silver instead of gold |
| Wrong geometry type | Stage 1 | All stages 2+ | Assumed spheres, paper uses rods |
| Wrong periodicity/coupling | Stage 2 | All stages 3+ | Missed inter-particle coupling |
| Wrong parameter range | Sweep stage | Later sweeps | Simulated 400-600nm, paper is 600-900nm |

**SupervisorAgent Decision Matrix**

| Suggestion Severity | Confidence | Backtrack Count | Decision |
|--------------------|------------|-----------------|----------|
| Critical | High | < MAX_BACKTRACKS | Accept backtrack |
| Critical | Low | Any | Ask user for confirmation |
| Significant | High | < MAX_BACKTRACKS | Accept backtrack |
| Significant | Low | Any | Ask user |
| Minor | Any | Any | Reject, handle locally |
| Any | Any | ≥ MAX_BACKTRACKS | Ask user (limit reached) |

**Backtrack Limits**

- `MAX_BACKTRACKS = 2` per reproduction
- If limit reached and another backtrack is needed → escalate to user
- Prevents infinite backtrack loops

---

### 14. ask_user Node

**Purpose**: Pause for user input and log decisions

**Triggers**:
- **Material validation checkpoint** (MANDATORY after Stage 0)
- Revision limits exceeded
- Ambiguous paper information
- Trade-off decisions needed
- Domain expertise required
- Backtrack approval needed

**Behavior**:
1. Set `awaiting_user_input = True`
2. Populate `pending_user_questions`
3. Pause graph execution
4. Wait for `user_responses` to be filled
5. **Log user interaction** to `user_interactions` list
6. Update `progress["user_interactions"]` for persistence
7. Resume to appropriate node

**User Interaction Logging**:
```python
def log_user_interaction(state, question, response, interaction_type):
    interaction = {
        "id": f"U{len(state['user_interactions']) + 1}",
        "timestamp": datetime.now().isoformat(),
        "interaction_type": interaction_type,  # e.g., "material_checkpoint", "trade_off_decision"
        "context": {
            "stage_id": state.get("current_stage_id"),
            "agent": "SupervisorAgent",
            "reason": state.get("supervisor_feedback", "User input required")
        },
        "question": question,
        "user_response": response,
        "impact": "",  # Filled after decision is applied
        "alternatives_considered": []
    }
    state["user_interactions"].append(interaction)
```

**Interaction Types**:
| Type | When Used |
|------|-----------|
| `material_checkpoint` | Mandatory Stage 0 approval |
| `clarification` | Ambiguous paper information |
| `trade_off_decision` | Accuracy vs runtime, 2D vs 3D, etc. |
| `parameter_confirmation` | Key parameter values |
| `stop_decision` | Whether to stop reproduction |
| `backtrack_approval` | Approving suggested backtrack |
| `context_overflow` | Context window exceeded, user decides recovery |
| `general_feedback` | Other user input |

#### CLI Interaction Model

ReproLab uses CLI-based prompts for user interaction. The `ask_user` node implementation:

**Interactive mode** (default):
- Prompts user in terminal with formatted questions
- Supports multi-line responses (press Enter twice to submit)
- Handles Ctrl+C gracefully with checkpoint save

**Non-interactive mode** (`REPROLAB_NON_INTERACTIVE=1`):
- Saves checkpoint immediately and exits
- Useful for batch/CI environments

**Timeout handling**:
- Configurable timeout (default: 24 hours)
- On timeout: saves checkpoint and exits (doesn't hang)

**Environment Variables**:

| Variable | Default | Description |
|----------|---------|-------------|
| `REPROLAB_USER_TIMEOUT_SECONDS` | 86400 (24h) | Timeout waiting for user response |
| `REPROLAB_NON_INTERACTIVE` | 0 | If "1", save checkpoint and exit instead of prompting |

**Resuming After Interruption**:

When the user is interrupted (Ctrl+C, timeout, or non-interactive mode), a checkpoint is saved:

```bash
# Resume from checkpoint
python -m src.graph --resume outputs/<paper_id>/checkpoints/checkpoint_<name>_latest.json
```

The checkpoint contains full state, so resumption continues exactly where paused.

---

### 14. generate_report Node (SupervisorAgent)

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
                                    │adapt_prompts│
                                    │(PromptAdapt)│
                                    └──────┬──────┘
                                           │
                                           ▼
                                    ┌─────────────┐
                              ┌─────│    plan     │◄────────────────┐
                              │     └──────┬──────┘                 │
                              │            │                        │
                              │            ▼                        │
                              │     ┌─────────────┐                 │
                              │     │ plan_review │                 │
                              │     └──────┬──────┘                 │
                              │            │                        │
                              │            ▼                        │
                              │     ┌─────────────┐                 │
                              │     │select_stage │                 │
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
                     ┌────────┴─►│   design    │◄─────────┐         │
                     │           │(Sim Designer)│         │         │
                     │           └──────┬──────┘          │         │
                     │                  │                 │         │
                     │                  ▼                 │         │
                     │           ┌─────────────┐          │         │
                     │           │design_review│          │         │
                     │           │ (design QA) │          │         │
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
                     │  │generate_code│◄──────┐   │                 │
                     │  │(CodeGenerator)│     │   │                 │
                     │  └──────┬──────┘       │   │                 │
                     │         │              │   │                 │
                     │         ▼              │   │                 │
                     │  ┌─────────────┐       │   │                 │
                     │  │ code_review │       │   │                 │
                     │  │  (code QA)  │       │   │                 │
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
                     │  │  run_code   │    │  ask_user   │◄─────┐   │
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
                     │  │physics_check│           │             │   │
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
                     │  │  analyze    │◄──────────┼─────┐       │   │
                     │  │(ResultsAnalyzer)│       │     │       │   │
                     │  └──────┬──────┘           │     │       │   │
                     │         │                  │     │       │   │
                     │         ▼                  │     │       │   │
                     │  ┌─────────────┐           │     │       │   │
                     │  │compare_CHECK│           │     │       │   │
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
                     │  │ supervisor  │───────────┴─────────────┘   │
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
| adapt_prompts | paper_text, paper_domain | prompt_adaptations |
| plan | paper_text, paper_domain | plan, assumptions, extracted_parameters, progress |
| plan_review | plan, progress | last_plan_review_verdict, replan_count, planner_feedback |
| select_stage | plan, progress, validation_hierarchy | current_stage_id, current_stage_type |
| design | plan, assumptions, reviewer_feedback | design_description, performance_estimate, new_assumptions |
| design_review | design_description, assumptions | last_design_review_verdict, reviewer_issues, design_revision_count |
| generate_code | design_description, reviewer_feedback | code |
| code_review | code, design_description | last_code_review_verdict, reviewer_issues, code_revision_count |
| run_code | code | stage_outputs, run_error, runtime_seconds |
| EXECUTION_CHECK | stage_outputs, run_error | execution_verdict, execution_failure_count |
| PHYSICS_SANITY | stage_outputs, code | physics_verdict, physics_failure_count, backtrack_suggestion |
| analyze | stage_outputs, paper_figures | analysis_summary, figure_comparisons, discrepancies_log |
| COMPARISON_CHECK | figure_comparisons, analysis_summary | comparison_verdict, analysis_revision_count |
| supervisor | progress, validation_hierarchy, user_responses | supervisor_verdict, progress updates, pending_user_questions |
| ask_user | pending_user_questions | user_responses, awaiting_user_input |
| generate_report | figure_comparisons, progress | report_conclusions, final_report_path |

### State Persistence Rules

1. **In-Memory State**: All fields in `ReproState` are in-memory during execution
2. **Disk Sync Points**:
   - `plan`, `assumptions`, `progress` are synced to JSON files at checkpoints
   - `figure_comparisons` aggregated into report at generate_report
   - `metrics` appended to log file at each agent call
3. **Canonical Sources**:
   - `plan["extracted_parameters"]` is canonical; `state.extracted_parameters` is synced view
   - `DISCREPANCY_THRESHOLDS` in state.py is canonical for threshold values

### State Validation Contract

Every node invocation is wrapped with validation to catch malformed state early:

1. **Pre-node validation**: `validate_state_for_node(state, node_name)` checks required fields
2. **Post-node validation**: `validate_state_transition(old_state, new_state, from_node, to_node)` validates mutations
3. **Failures are structured**: Missing fields surface as structured `ValidationError`, not cryptic `KeyError`s

This wrapper is **mandatory** for all nodes and must not be bypassed:

```python
def run_node_with_validation(state: ReproState, node_name: str, node_fn: Callable) -> Dict[str, Any]:
    """Wrapper that validates state before and after node execution."""
    # Pre-validation
    missing = validate_state_for_node(state, node_name)
    if missing:
        raise ValidationError(f"Missing fields for {node_name}: {missing}")
    
    # Execute node
    old_state = state.copy()
    result = node_fn(state)
    
    # Post-validation (for transitions)
    # ...
    
    return result
```

### Checkpoint Trigger Points

| Checkpoint Name | Trigger Location | Contents Saved |
|-----------------|------------------|----------------|
| `after_plan` | After plan node | Full state with plan, assumptions |
| `after_stage0_user_confirm` | After user confirms materials | State + user confirmation |
| `after_stage_N_complete` | After each supervisor approval | Full state with stage progress |
| `before_ask_user` | Before ask_user node | Full state for resume |
| `final_report` | After generate_report | Full state + report path |

## State Persistence

### IMPORTANT: Dual Checkpointing Strategy

The system uses TWO checkpointing mechanisms that serve different purposes:

#### 1. LangGraph's MemorySaver (Source of Truth for Execution)

```python
checkpointer = MemorySaver()
workflow.compile(checkpointer=checkpointer, interrupt_before=["ask_user"])
```

**Purpose**: Graph execution state management
**Used for**: Resuming graph execution after interrupts (e.g., `ask_user`)
**Scope**: Complete `ReproState` including internal workflow state
**Managed by**: LangGraph runtime automatically

#### 2. Disk JSON Files (Artifacts for Humans/Debugging)

```
outputs/<paper_id>/
├── _artifact_plan.json           # Underscore prefix = artifact (not for execution)
├── _artifact_assumptions.json
├── _artifact_progress.json
└── checkpoints/checkpoint_*.json  # LangGraph checkpoints (resumable)
```

**Purpose**: Human-readable artifacts, debugging, manual inspection
**Used for**: Reviewing what happened, debugging failures, archival
**Scope**: Selected state fields (plan, assumptions, progress)
**Managed by**: `src/persistence.py` module with safety guards

**File Naming Convention**: Artifact files use `_artifact_` prefix to make it 
visually obvious they are not for loading during execution.

#### Why This Matters (Potential Sync Issues)

These two systems can get out of sync:

| Scenario | LangGraph State | Disk JSON | Risk |
|----------|-----------------|-----------|------|
| Normal operation | Current | Slightly behind | Low - disk updated at checkpoints |
| Crash mid-node | Last checkpoint | Further behind | Medium - disk may be stale |
| Manual JSON edit | Unchanged | Modified | **HIGH** - systems diverge |

#### Rules to Prevent Issues

1. **NEVER read from disk JSONs for execution decisions**
   - Always use `state` passed by LangGraph
   - Disk JSONs are OUTPUT only, not INPUT

2. **Treat disk JSONs as artifacts**
   - For debugging: "What did the plan look like at stage 2?"
   - For review: "What assumptions were made?"
   - NOT for: "What should happen next?"

3. **LangGraph state is authoritative**
   - If resuming after crash, resume from LangGraph checkpoint
   - Disk JSONs may be regenerated from state if needed

4. **save_checkpoint() is for humans, not the graph**
   - Called at strategic points for human inspection
   - Does not affect graph execution flow

#### Enforcement via `src/persistence.py`

The `src/persistence.py` module enforces these rules programmatically:

```python
from src.persistence import (
    save_artifact,              # ✅ Safe: writing artifacts
    read_artifact_for_debugging, # ⚠️ Warning: for debugging only
    load_plan_from_disk,        # ❌ FORBIDDEN: always raises error
)

# ✅ CORRECT: Save artifacts for human review
save_artifact(state["plan"], path, "plan")

# ⚠️ FOR DEBUGGING ONLY: Reads with warning
plan = read_artifact_for_debugging(path, caller="debug_session")
# → Prints warning: "This data may be out of sync..."

# ❌ FORBIDDEN: These ALWAYS raise DiskReadForbiddenError
plan = load_plan_from_disk("paper_123")  # Raises error!
# → "FORBIDDEN: Cannot load plan from disk for execution..."
```

**Why trap functions exist**: If someone accidentally tries to load state from
disk during execution, they get a clear error with instructions instead of
subtle state divergence bugs that are hard to debug.

**Audit trail**: All artifact reads are logged to `_artifact_access_log.txt`
in the paper's output directory for debugging and auditing.

### Checkpointing

The graph supports checkpointing at key points:
- `after_plan`: After PlannerAgent completes the reproduction plan
- `after_stage_complete`: After each stage completes (supervisor node)
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
   - `"planning"` → Resume at plan node
   - `"design"` → Resume at design node
   - `"running"` → Resume at run_code node
   - `"analysis"` → Resume at analyze node
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

State is mirrored to artifact files (managed by `src/persistence.py`):
```
outputs/<paper_id>/
├── _artifact_plan.json
├── _artifact_assumptions.json
├── _artifact_progress.json
├── _artifact_access_log.txt       # Audit trail for debugging reads
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

## Context Window Management

### Overview

Claude Opus 4.5 has a 200K token context window, but we reserve tokens for prompts, state context, and response generation. Long papers, accumulated feedback, or many revision cycles can exceed safe limits.

**Key functions** (defined in `schemas/state.py`):
- `check_context_before_node(state, node_name)` - Main entry point
- `estimate_context_for_node(state, node_name)` - Calculate tokens
- `get_context_recovery_actions(node_name, tokens, state)` - Recovery options

### Integration Points

**Every LLM-calling node MUST call `check_context_before_node()` at the start:**

```python
from schemas.state import check_context_before_node

def design_node(state):
    """Example: SimulationDesignerAgent node with context management."""
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Check context limits before LLM call
    # ═══════════════════════════════════════════════════════════════════
    check = check_context_before_node(state, "design")
    
    if check["escalate"]:
        # Context overflow - need user guidance
        return {
            "pending_user_questions": [check["user_question"]],
            "awaiting_user_input": True,
            "ask_user_trigger": "context_overflow",
            "last_node_before_ask_user": "design",
        }
    
    if check["state_updates"]:
        # Auto-recovery was applied (e.g., feedback truncation)
        # Merge updates into state
        state = {**state, **check["state_updates"]}
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Proceed with normal LLM call
    # ═══════════════════════════════════════════════════════════════════
    prompt = build_prompt("simulation_designer", state)
    response = llm.invoke(prompt, tools=[design_schema])
    
    return {"design_description": response.content}
```

### Which Nodes Need Context Checks

| Node | Needs Check | Reason |
|------|-------------|--------|
| `plan` | ✅ Yes | Receives full paper text (largest input) |
| `adapt_prompts` | ✅ Yes | Receives full paper text |
| `design` | ✅ Yes | Accumulates feedback over revisions |
| `generate_code` | ✅ Yes | Design + feedback + code grows |
| `plan_review` | ✅ Yes | Full plan context |
| `design_review` | ✅ Yes | Design description |
| `code_review` | ✅ Yes | Code + design |
| `analyze` | ✅ Yes | Multiple figures + outputs |
| `supervisor` | ⚠️ Minimal | Receives summaries, not raw data |
| `run_code` | ❌ No | Not an LLM call |
| `select_stage` | ❌ No | Pure logic, no LLM |
| `material_checkpoint` | ❌ No | User interaction, no LLM |

### Recovery Priority Order

When context exceeds limits, recovery actions are attempted in this order:

| Priority | Action | Risk | Auto-Applied? |
|----------|--------|------|---------------|
| 1 | Summarize feedback | Low | ❌ Needs LLM |
| 2 | Truncate feedback to last 2000 chars | Low | ✅ Yes |
| 3 | Use Methods section only | Medium | ❌ No |
| 4 | Clear non-critical working fields | Medium | ❌ No |
| 5 | Escalate to user | None | Always available |

### Context Budget Tracking

The `context_budget` field in state tracks estimated tokens across the workflow:

```python
# After each LLM call, update context budget tracking
state["metrics"]["agent_calls"].append({
    "agent": "SimulationDesignerAgent",
    "timestamp": datetime.now().isoformat(),
    "input_tokens": response.usage.input_tokens,
    "output_tokens": response.usage.output_tokens,
    "context_estimate": check["estimation"]["estimated_tokens"],
})
```

### Handling Context Overflow Escalation

When `check_context_before_node()` returns `escalate=True`:

1. Set `ask_user_trigger = "context_overflow"`
2. Present user with recovery options from `check["available_actions"]`
3. User selects action (summarize, truncate, skip stage, stop)
4. Apply selected action and resume

```python
# In ask_user node, handle context_overflow trigger
if state.get("ask_user_trigger") == "context_overflow":
    user_choice = state["user_responses"].get("context_recovery")
    
    if user_choice == "summarize_feedback":
        # Use LLM to summarize, then retry
        summary = summarize_feedback(state["reviewer_feedback"])
        return {"reviewer_feedback": summary, "awaiting_user_input": False}
    
    elif user_choice == "truncate_paper":
        # Keep only Methods section
        state["paper_text"] = extract_methods_section(state["paper_text"])
        return {"paper_text": state["paper_text"], "awaiting_user_input": False}
    
    elif user_choice == "skip_stage":
        # Mark current stage as blocked
        update_progress_stage_status(state, state["current_stage_id"], "blocked",
            summary="Skipped due to context limits")
        return {"awaiting_user_input": False}
```

### Loop Context Estimation

During revision loops (design→review→revise), context grows. Use these helpers:

```python
from schemas.state import check_loop_context_status

# Before starting a revision iteration
status = check_loop_context_status(
    loop_type="design",
    iteration=state["design_revision_count"],
    current_artifact_chars=len(state.get("design_description", "")),
    feedback_history_chars=len(state.get("reviewer_feedback", ""))
)

if status["should_summarize"]:
    # Summarize feedback before continuing
    state["reviewer_feedback"] = summarize_feedback(state["reviewer_feedback"])
```

---

## Error Handling

### Simulation Errors

The system tracks two distinct types of code-related failures:

1. **`code_revision_count`**: Incremented when CodeReviewerAgent requests changes (before execution)
2. **`execution_failure_count`**: Incremented when simulation runs but crashes/fails at runtime

This distinction is important because:
- Code revisions from review are typically style/correctness issues fixable by the LLM
- Execution failures may indicate fundamental issues (memory, timeout, numerical instability)
- Different escalation paths may be appropriate

```python
# ExecutionValidatorAgent determines if execution failed
if execution_failed:
    # Increment execution failure count (distinct from code_revision_count)
    state["execution_failure_count"] += 1
    
    max_failures = state["runtime_config"].get("max_execution_failures", MAX_EXECUTION_FAILURES)
    
    if state["execution_failure_count"] >= max_failures:
        # Escalate to user - execution failures may need human intervention
        state["pending_user_questions"].append(
            f"Simulation crashed {state['execution_failure_count']} times. "
            f"Error: {state['run_error']}. This may indicate:\n"
            f"1. Memory issues (try reducing resolution)\n"
            f"2. Numerical instability (check source/geometry)\n"
            f"3. Meep version incompatibility\n"
            f"How should we proceed?"
        )
        return "ask_user"
    
    # Try regenerating code with execution error context
    state["reviewer_feedback"] = f"Simulation execution failed: {state['run_error']}"
    return "generate_code"
```

**Note**: Both `code_revision_count` and `execution_failure_count` are reset per stage. 
Additionally, `execution_failure_count` resets when user intervenes via ask_user.
Global tracking via `total_execution_failures` preserves the full count for metrics/reporting.
See "Counter Scoping and Reset Behavior" section below for full details.

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

### Error Recovery Matrix

This matrix defines the system behavior for various error scenarios and edge cases:

| Scenario | Behavior | Limit | Escalation |
|----------|----------|-------|------------|
| **Simulation execution fails (crash/timeout)** | Increment `execution_failure_count`. Regenerate code with error context. After limit, escalate to user. | `max_execution_failures` (default: 2) | ask_user |
| **Code review rejects code** | Increment `code_revision_count`. Regenerate with reviewer feedback. After limit, escalate. | `MAX_CODE_REVISIONS` (3) | ask_user |
| **physics_check fails repeatedly** | After 2 failures at same stage, escalate to supervisor with `physics_stuck` flag. Supervisor can try alternative approach or ask user. | 2 per stage | supervisor → ask_user |
| **User doesn't respond to ask_user** | Timeout after configurable period (default: 24 hours). Auto-save checkpoint, pause workflow. Can be resumed later. | 24 hours (default) | Checkpoint + Pause |
| **Total runtime exceeded** | Hard abort after `max_total_runtime_hours`. Save checkpoint, generate partial report with whatever results are available. | 8 hours (default) | Partial report |
| **Consecutive stage failures** | After 2 consecutive stage failures (different stages), trigger replanning via supervisor | 2 consecutive | supervisor → plan |
| **Memory exhaustion during simulation** | Capture error, increment `execution_failure_count`, suggest resolution reduction or cell size adjustment. | `max_execution_failures` | ask_user |
| **LLM context overflow** | Auto-recovery attempts: 1) truncate feedback, 2) summarize paper. If insufficient, escalate to user with options. | `safe_paper_tokens` (140K) | ask_user |
| **LLM rate limit or timeout** | Retry with exponential backoff (1s, 2s, 4s, 8s, 16s). After 5 retries, save checkpoint and pause. | 5 retries | Checkpoint + Pause |
| **Invalid JSON from LLM** | Retry same call up to 3 times. If still failing, escalate to user with raw output for debugging. | 3 retries | ask_user |
| **File I/O errors** | Log error, attempt alternate paths. If critical file (checkpoint, output), escalate immediately. | N/A | ask_user |

### Counter Scoping and Reset Behavior

Understanding when counters reset is critical for error recovery:

| Counter | Scope | Reset Conditions | Global Tracking |
|---------|-------|------------------|-----------------|
| `execution_failure_count` | Per-stage | Stage completes successfully, User intervenes, New stage starts | `total_execution_failures` (never resets) |
| `code_revision_count` | Per-stage | New stage starts | No |
| `design_revision_count` | Per-stage | New stage starts | No |
| `physics_failure_count` | Per-stage | New stage starts | No |

**Why per-stage scoping?**
- Prevents one problematic stage from poisoning the entire workflow
- Each new stage gets a fresh chance
- User intervention gives the system another opportunity with guidance

**Counter reset implementation:**
```python
def reset_stage_counters(state):
    """Reset per-stage counters when moving to a new stage."""
    state["execution_failure_count"] = 0
    state["code_revision_count"] = 0
    state["design_revision_count"] = 0
    state["physics_failure_count"] = 0
    # Note: total_execution_failures is NOT reset

def handle_user_intervention(state):
    """Reset counters after user provides guidance."""
    # User guidance = fresh start for this stage
    state["execution_failure_count"] = 0
    # code_revision_count may or may not reset depending on intervention type
```

### Context Overflow Recovery

Long papers (100K+ chars) combined with accumulated feedback during revision loops can 
exceed the LLM's context window, causing cryptic API failures. The system includes 
proactive context management to prevent and recover from overflow situations.

#### Context Estimation

Before each LLM call, estimate context size:

```python
from schemas.state import check_context_before_node

def some_node(state):
    # Check context before making LLM calls
    check = check_context_before_node(state, "design")
    
    if check["escalate"]:
        # Context too large, user intervention needed
        return {
            "pending_user_questions": [check["user_question"]],
            "awaiting_user_input": True,
        }
    
    if check["state_updates"]:
        # Auto-recovery was applied, merge updates
        state = {**state, **check["state_updates"]}
    
    # Proceed with normal node logic...
```

#### Recovery Actions (Priority Order)

| Priority | Action | Risk | Requires LLM | Description |
|----------|--------|------|--------------|-------------|
| 1 | `summarize_feedback` | Low | Yes | Condense reviewer feedback to ~500 chars using LLM |
| 2 | `truncate_feedback` | Low | No | Keep only last 2000 chars of feedback |
| 3 | `truncate_paper_to_methods` | Medium | No | Keep only Methods section of paper |
| 4 | `clear_working_fields` | Medium | No | Clear regenerable fields like `analysis_summary` |
| 5 | `escalate_to_user` | None | No | Present options to user for decision |

#### User Escalation

When automatic recovery is insufficient, the user sees:

```
Context overflow detected in design.

**Current estimate:** 185,000 tokens
**Safe limit:** 140,000 tokens  
**Over by:** 45,000 tokens

Available recovery options:
1. Summarize reviewer feedback (12,500 chars → ~500 chars) (saves ~3,000 tokens, risk: low)
2. Keep only Methods section (95,000 chars → ~20K chars) (saves ~18,750 tokens, risk: medium)
3. Skip this stage and continue
4. Stop reproduction

Which option should we use?
```

#### Key Functions

| Function | Purpose |
|----------|---------|
| `estimate_context_for_node()` | Estimate tokens for a specific node |
| `get_context_recovery_actions()` | Get available recovery actions (doesn't mutate state) |
| `check_context_before_node()` | Main entry point - check + auto-recover if possible |
| `ContextOverflowError` | Exception raised when recovery fails |

#### Design Principles

1. **Functions return actions, don't mutate state** - Follows LangGraph pattern where 
   nodes apply state changes through return values
2. **Low-risk first** - Automatic recovery only attempts low-risk actions
3. **User always last** - Escalation is the final fallback
4. **Builds on existing infrastructure** - Uses existing `create_feedback_summary_prompt()` 
   and loop context estimation functions
```

### Recovery Implementation

```python
def handle_physics_stuck(state):
    """Handle repeated physics validation failures."""
    physics_failures = state.get("physics_failure_count", 0)
    
    if physics_failures >= 2:
        # Escalate to supervisor with flag
        state["physics_stuck"] = True
        state["supervisor_hint"] = (
            f"Physics validation has failed {physics_failures} times for stage "
            f"{state['current_stage_id']}. Consider: (1) simplifying geometry, "
            f"(2) checking material models, (3) asking user for guidance."
        )
        return "supervisor"
    
    # Otherwise, increment and retry
    state["physics_failure_count"] = physics_failures + 1
    return "generate_code"


def handle_timeout(state, config):
    """Handle various timeout scenarios."""
    total_runtime = state.get("total_runtime_seconds", 0)
    max_runtime = config.max_total_runtime_hours * 3600
    
    if total_runtime >= max_runtime:
        # Save final checkpoint
        save_checkpoint(state, "timeout_abort")
        
        # Generate partial report
        state["should_stop"] = True
        state["stop_reason"] = f"Total runtime budget exceeded ({config.max_total_runtime_hours}h)"
        return "generate_report"
    
    return None  # Continue normally
```

## Debug Mode / Quick Check

When debugging issues or validating a new paper setup, you can run a minimal diagnostic pass instead of the full reproduction workflow.

### What Debug Mode Does

| Feature | Full Mode | Debug Mode |
|---------|-----------|------------|
| Resolution | Per-design optimal | Minimum viable (λ/8) |
| Stages | All planned | Stage 0 + minimal Stage 1 only |
| Sweeps | Full parameter range | Single point per sweep |
| Runtime budget | 8 hours | 30 minutes |
| Output | Full reports | Diagnostic summary |

### When to Use Debug Mode

1. **New paper setup**: Verify paper loading, figure extraction, parameter parsing
2. **Material debugging**: Test if material models are working correctly
3. **Geometry debugging**: Check if structure is being created as expected
4. **Quick sanity check**: Verify basic simulation runs before committing to full reproduction
5. **After errors**: Diagnose why a full run failed

### Debug Mode Configuration

Enable debug mode via `RuntimeConfig`:

```python
from schemas.state import RuntimeConfig

debug_config = RuntimeConfig(
    max_total_runtime_hours=0.5,  # 30 minutes max
    max_stage_runtime_minutes=10,  # 10 minutes per stage
    debug_mode=True,  # Enable debug mode
    debug_resolution_factor=0.5,  # Half normal resolution
    debug_max_stages=2,  # Only Stage 0 and Stage 1
)
```

### Debug Mode Outputs

Debug mode produces diagnostic outputs:

```
outputs/<paper_id>/debug/
├── geometry_check.png         # Visualization of simulation geometry
├── material_check.png         # Plot of material optical properties
├── source_spectrum.png        # Source spectrum verification
├── monitor_positions.png      # Monitor placement visualization
├── debug_summary.json         # Quick diagnostic summary
└── debug_log.txt              # Detailed execution log
```

### Diagnostic Summary Format

```json
{
  "paper_id": "smith2023_plasmon",
  "debug_run_timestamp": "2025-11-30T10:15:00Z",
  "total_runtime_seconds": 145,
  
  "paper_loading": {
    "status": "success",
    "text_chars": 45000,
    "figures_found": 6,
    "figures_loaded": 6
  },
  
  "stage0_material_validation": {
    "status": "success",
    "materials_tested": ["aluminum", "glass"],
    "issues": []
  },
  
  "stage1_minimal": {
    "status": "success",
    "simulation_ran": true,
    "output_files_created": ["debug_spectrum.csv", "debug_spectrum.png"],
    "resonance_found": true,
    "resonance_wavelength_nm": 520
  },
  
  "geometry_check": {
    "structures_created": 1,
    "structure_types": ["cylinder"],
    "dimensions_match_design": true
  },
  
  "recommendations": [
    "Material validation passed - proceed to full run",
    "Consider increasing resolution for final run (current: 25, recommended: 50)"
  ]
}
```

### Running Debug Mode

```python
from src.graph import create_repro_graph
from src.paper_loader import load_paper_from_markdown
from schemas.state import RuntimeConfig

# Load paper
paper_input = load_paper_from_markdown(...)

# Create graph with debug config
debug_config = RuntimeConfig(debug_mode=True, max_total_runtime_hours=0.5)
app = create_repro_graph(runtime_config=debug_config)

# Run debug pass
result = app.invoke(paper_input)

# Check diagnostic summary
print(result["debug_summary"])
```

### Debug Mode Limitations

- Does NOT validate full quantitative accuracy
- Does NOT run complete parameter sweeps
- May miss issues that only appear at high resolution
- Should be followed by full run for actual reproduction

## Configuration

### Environment Variables

```bash
# LLM Configuration
ANTHROPIC_API_KEY=your-key

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

**v1 Configuration: Claude Opus 4.5 for all agents**

For the initial version, we use Claude Opus 4.5 (`claude-opus-4-20250514`) consistently across all agents. This provides:
- Best-in-class reasoning for complex physics
- Excellent code generation quality
- Strong vision capabilities for figure comparison
- Consistent behavior across all agents

```python
from langchain_anthropic import ChatAnthropic

# v1: Single model for all agents (Claude Opus 4.5)
MODEL = "claude-opus-4-20250514"

# All agents use the same model
prompt_adaptor_llm = ChatAnthropic(model=MODEL, temperature=0)
planner_llm = ChatAnthropic(model=MODEL, temperature=0)
designer_llm = ChatAnthropic(model=MODEL, temperature=0)
code_generator_llm = ChatAnthropic(model=MODEL, temperature=0)
reviewer_llm = ChatAnthropic(model=MODEL, temperature=0)
execution_validator_llm = ChatAnthropic(model=MODEL, temperature=0)
physics_sanity_llm = ChatAnthropic(model=MODEL, temperature=0)
analyzer_llm = ChatAnthropic(model=MODEL, temperature=0)
comparison_validator_llm = ChatAnthropic(model=MODEL, temperature=0)
supervisor_llm = ChatAnthropic(model=MODEL, temperature=0)
```

**Future: Per-agent model selection (v2)**

Once the base system is validated, we can optimize costs by using different models:
- Cheaper models (Claude Sonnet, GPT-4o-mini) for focused validation tasks
- Premium models (Opus, GPT-4o) for complex reasoning (planning, design, analysis)
- Multi-model consensus for critical decisions

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
- Feedback is passed via state fields (e.g., `reviewer_feedback`, `reviewer_issues`)

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
    ↓ sets last_reviewer_verdict="needs_revision"
CodeGeneratorAgent (attempt 2)
    ↓ receives: fresh prompt + state including reviewer_feedback
    ↓ uses feedback to improve code
CodeReviewerAgent
    ↓ reviews improved code
    ↓ sets last_reviewer_verdict="approve"
```

## Prompt Construction and Injection

Agent prompts are constructed at runtime by the `src/prompts.py` module. This ensures:
1. **Single source of truth** for constants (thresholds, limits)
2. **Dynamic state injection** (paper text, figures, current stage)
3. **Consistent prompt structure** across all agents

### How Prompts Are Built

```python
from src.prompts import build_prompt

# For each agent call:
prompt = build_prompt(
    agent_name="planner",     # Which agent
    state=current_state,       # Current ReproState
    include_global_rules=True  # Prepend global rules
)

response = llm.invoke(prompt)
```

### Injection Process

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROMPT CONSTRUCTION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Load global_rules.md                                        │
│     └─ Replace {THRESHOLDS_TABLE} → actual table from state.py  │
│     └─ Replace {MAX_DESIGN_REVISIONS} → "3"                     │
│                                                                 │
│  2. Load agent-specific prompt (e.g., planner_agent.md)         │
│     └─ Replace placeholders with constants                      │
│                                                                 │
│  3. Append state context (agent-specific)                       │
│     └─ PlannerAgent: paper_text, paper_figures                  │
│     └─ DesignerAgent: current_stage, assumptions, feedback      │
│     └─ AnalyzerAgent: stage_outputs, target_figures             │
│                                                                 │
│  4. Return complete prompt                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Placeholders

Prompts use placeholders that are replaced at runtime:

| Placeholder | Source | Used In |
|-------------|--------|---------|
| `{THRESHOLDS_TABLE}` | `state.py:DISCREPANCY_THRESHOLDS` | global_rules.md, results_analyzer_agent.md |
| `{MAX_DESIGN_REVISIONS}` | `state.py:MAX_DESIGN_REVISIONS` | Anywhere revision limits mentioned |
| `{MAX_REPLANS}` | `state.py:MAX_REPLANS` | Workflow decision prompts |

### State Context by Agent (Detailed)

Each agent receives **only what it needs** - not full repo access. This section documents exactly what context is injected for each agent.

---

#### PromptAdaptorAgent Context

**Purpose**: Quick scan of paper to customize agent prompts for the domain.

**Receives**:
| Field | Source | Notes |
|-------|--------|-------|
| `paper_text_summary` | Extracted from `state["paper_text"]` | Abstract + Methods section (critical for optics papers) |
| `paper_domain` | `state["paper_domain"]` | e.g., "plasmonics", "photonic_crystals" |
| `available_agents` | Hardcoded list | Names and roles of all agents |
| `available_prompts` | `prompts/*.md` file list | What can be adapted |

**Does NOT receive**: Full paper text (Results/Discussion), figures, any simulation-related state.

**Rationale**: Needs Abstract for high-level domain understanding AND Methods section for critical details:
- Material data sources (Palik vs Johnson-Christy, etc.)
- Exact geometry specifications and fabrication details
- Simulation parameters mentioned by original authors
- Measurement techniques that affect interpretation

**Implementation note**: Extract Methods section by looking for common headers ("Methods", "Experimental", "Materials and Methods", "Simulation Details"). If not found, fall back to first ~15,000 chars.

---

#### PlannerAgent Context

**Purpose**: Full paper analysis, parameter extraction, stage planning.

**Receives**:
| Field | Source | Notes |
|-------|--------|-------|
| `paper_id` | `state["paper_id"]` | Identifier |
| `paper_title` | `state["paper_title"]` | From paper metadata |
| `paper_domain` | `state["paper_domain"]` | Domain classification |
| `paper_text` | `state["paper_text"]` | **Full** paper text (markdown) |
| `paper_figures` | `state["paper_figures"]` | List of {id, description, image_path} |
| `supplementary_text` | `state["supplementary_text"]` | If available |
| `supplementary_figures` | `state["supplementary_figures"]` | If available |
| `prompt_adaptations` | `state["prompt_adaptations"]` | From PromptAdaptorAgent |
| `user_interactions` | `state["user_interactions"]` | **If replanning** - user corrections to apply |
| `supervisor_feedback` | `state["supervisor_feedback"]` | **If replanning** - why we're replanning |
| `replan_reason` | Computed from context | Summary of what triggered replanning |

**Does NOT receive**: Any simulation outputs, code, progress, validation state.

**Rationale**: Needs complete paper information to create comprehensive plan. When replanning, also needs user corrections to avoid repeating errors.

---

#### SimulationDesignerAgent Context

**Purpose**: Design simulation for current stage based on plan.

**Receives**:
| Field | Source | Notes |
|-------|--------|-------|
| `current_stage_id` | `state["current_stage_id"]` | Which stage we're designing |
| `current_stage` | `state["plan"]["stages"][current]` | Stage requirements from plan |
| `assumptions` | `state["assumptions"]` | All documented assumptions |
| `extracted_parameters` | `state["extracted_parameters"]` | Parameter values with sources |
| `reviewer_feedback` | `state["reviewer_feedback"]` | **Only if revision** - previous issues |
| `revision_count` | `state["design_revision_count"]` | How many revisions so far |
| `paper_domain` | `state["paper_domain"]` | For domain-specific defaults |
| `available_materials` | `materials/index.json` | What materials are available |
| `user_interactions` | `state["user_interactions"]` | **If available** - user corrections that override parameters |
| `supervisor_feedback` | `state["supervisor_feedback"]` | **If available** - supervisor guidance on design issues |

**Does NOT receive**: Full paper text, figures, other stages, simulation code, outputs.

**Rationale**: Focused on current stage design. Paper details already extracted to plan. User corrections must be checked before using extracted parameters.

---

#### CodeGeneratorAgent Context

**Purpose**: Generate Python+Meep code from approved design.

**Receives**:
| Field | Source | Notes |
|-------|--------|-------|
| `design_description` | `state["design_description"]` | Approved design spec |
| `performance_estimate` | `state["performance_estimate"]` | Expected runtime, memory |
| `reviewer_feedback` | `state["reviewer_feedback"]` | **Only if revision** |
| `revision_count` | `state["code_revision_count"]` | How many revisions so far |
| `stage_id` | `state["current_stage_id"]` | For output naming |
| `paper_id` | `state["paper_id"]` | For output naming |
| `output_dir` | Computed | Where to save outputs |

**Does NOT receive**: Full paper, plan details, assumptions, figures, previous code.

**Rationale**: Design spec contains everything needed. Doesn't need paper interpretation.

---

#### CodeReviewerAgent Context

**Purpose**: Review design OR code before execution.

**For Design Review**:
| Field | Source | Notes |
|-------|--------|-------|
| `design_description` | `state["design_description"]` | Design to review |
| `current_stage` | `state["plan"]["stages"][current]` | Stage requirements |
| `assumptions` | `state["assumptions"]` | Check consistency |
| `paper_domain` | `state["paper_domain"]` | Domain-specific checks |

**For Code Review**:
| Field | Source | Notes |
|-------|--------|-------|
| `code` | `state["code"]` | Code to review |
| `design_description` | `state["design_description"]` | What code should implement |
| `expected_outputs` | From design | What files should be created |

**Does NOT receive**: Full paper, figures, previous outputs, other stages.

**Rationale**: Focused review on specific artifact. Doesn't need full context.

---

#### ExecutionValidatorAgent Context

**Purpose**: Verify simulation ran correctly.

**Receives**:
| Field | Source | Notes |
|-------|--------|-------|
| `stdout` | `state["stage_outputs"]["stdout"]` | Simulation output |
| `stderr` | `state["stage_outputs"]["stderr"]` | Error output |
| `exit_code` | `state["stage_outputs"]["exit_code"]` | 0 = success |
| `output_files` | `state["stage_outputs"]["files"]` | List of created files |
| `expected_outputs` | From design | What was expected |
| `runtime_seconds` | `state["stage_outputs"]["runtime"]` | How long it took |

**Does NOT receive**: The code itself, paper, figures, design details.

**Rationale**: Only checks execution success, not correctness of design.

---

#### PhysicsSanityAgent Context

**Purpose**: Validate physical reasonableness of results.

**Receives**:
| Field | Source | Notes |
|-------|--------|-------|
| `output_data_files` | `state["stage_outputs"]["files"]` | CSV/data files to analyze |
| `simulation_code` | `state["code"]` | To understand what was computed |
| `paper_domain` | `state["paper_domain"]` | Domain-specific physics checks |
| `current_stage` | `state["plan"]["stages"][current]` | What physics is expected |

**Does NOT receive**: Paper text, figures, previous stages' data.

**Rationale**: Checks physics (T≤1, conservation laws) independent of paper comparison.

---

#### ResultsAnalyzerAgent Context

**Purpose**: Compare simulation results to paper figures.

**Receives**:
| Field | Source | Notes |
|-------|--------|-------|
| `output_files` | `state["stage_outputs"]["files"]` | Plots and data from simulation |
| `target_figures` | `state["paper_figures"][target_ids]` | **Only** the figures this stage targets |
| `digitized_data` | From target `paper_figures` entries (`figure["digitized_data_path"]`) | Optional CSV for quantitative comparison |
| `validation_criteria` | `state["plan"]["stages"][current]` | What to check |
| `stage_id` | `state["current_stage_id"]` | For reporting |
| `discrepancy_thresholds` | Constants | Classification thresholds |

**Does NOT receive**: Full paper text, non-target figures, code, design.

**Rationale**: Focused comparison. Paper details already encoded in validation criteria.

---

#### ComparisonValidatorAgent Context

**Purpose**: Validate that ResultsAnalyzer's comparison is accurate.

**Receives**:
| Field | Source | Notes |
|-------|--------|-------|
| `figure_comparisons` | `state["figure_comparisons"]` | What Analyzer produced |
| `analysis_summary` | `state["analysis_summary"]` | Analyzer's conclusions |
| `discrepancies` | From figure_comparisons | Logged discrepancies |
| `target_figures` | `state["paper_figures"][target_ids]` | To verify comparison |
| `output_plots` | `state["stage_outputs"]["plots"]` | Simulation plots |

**Does NOT receive**: Paper text, code, design, other stages.

**Rationale**: QA on comparison accuracy, not redoing the analysis.

---

#### SupervisorAgent Context

**Purpose**: Big-picture oversight, decision making.

**Receives** (most comprehensive view):
| Field | Source | Notes |
|-------|--------|-------|
| `plan_summary` | Summarized from `state["plan"]` | Stage names, targets, status |
| `progress` | `state["progress"]` | All stage statuses |
| `validation_hierarchy` | `state["validation_hierarchy"]` | What's validated |
| `figure_comparisons` | `state["figure_comparisons"]` | All comparisons so far |
| `current_stage_summary` | Summarized | What just happened |
| `runtime_budget` | `state["runtime_budget_remaining_seconds"]` | Time left |
| `revision_counts` | Various `state` fields | How many revisions used |
| `backtrack_suggestion` | `state["backtrack_suggestion"]` | If any agent suggested |
| `systematic_discrepancies` | `state["systematic_discrepancies_identified"]` | Known shifts |
| `user_responses` | `state["user_responses"]` | Current user answers (question→response) |
| `user_interactions` | `state["user_interactions"]` | Full log of user decisions |
| `pending_user_questions` | `state["pending_user_questions"]` | Outstanding questions |
| `ask_user_trigger` | `state["ask_user_trigger"]` | What triggered last ask_user (e.g., "material_checkpoint") |
| `last_node_before_ask_user` | `state["last_node_before_ask_user"]` | Which node triggered the interrupt |

**Does NOT receive**: Full paper text, raw code, raw data files.

**Rationale**: Strategic oversight needs summary, not raw details. User feedback is critical for making decisions after ask_user resumes.

---

### Context Building Implementation

```python
def build_agent_context(state: ReproState, agent: str) -> Dict[str, Any]:
    """Build context dictionary for a specific agent."""
    
    if agent == "PlannerAgent":
        context = {
            "paper_id": state["paper_id"],
            "paper_title": state["paper_title"],
            "paper_domain": state["paper_domain"],
            "paper_text": state["paper_text"],  # Full
            "paper_figures": state["paper_figures"],
            "supplementary_text": state.get("supplementary_text"),
            "supplementary_figures": state.get("supplementary_figures"),
            "prompt_adaptations": state.get("prompt_adaptations", []),
        }
        # Add user feedback if replanning
        if state.get("replan_count", 0) > 0:
            context["user_interactions"] = state.get("user_interactions", [])
            context["supervisor_feedback"] = state.get("supervisor_feedback")
            context["replan_reason"] = _compute_replan_reason(state)
        return context
    
    elif agent == "SimulationDesignerAgent":
        current_stage = get_current_stage(state)
        context = {
            "current_stage_id": state["current_stage_id"],
            "current_stage": current_stage,
            "assumptions": state["assumptions"],
            "extracted_parameters": state["extracted_parameters"],
            "reviewer_feedback": state.get("reviewer_feedback"),  # May be None
            "revision_count": state["design_revision_count"],
            "paper_domain": state["paper_domain"],
            "available_materials": load_materials_index(),
        }
        # Always include user interactions - they may contain parameter corrections
        if state.get("user_interactions"):
            context["user_interactions"] = state["user_interactions"]
        if state.get("supervisor_feedback"):
            context["supervisor_feedback"] = state["supervisor_feedback"]
        return context
    
    elif agent == "ResultsAnalyzerAgent":
        target_ids = get_target_figure_ids(state)
        target_figures = [f for f in state["paper_figures"] if f["id"] in target_ids]
        digitized_lookup = {
            fig["id"]: fig["digitized_data_path"]
            for fig in target_figures
            if fig.get("digitized_data_path")
        }
        return {
            "output_files": state["stage_outputs"]["files"],
            "target_figures": target_figures,
            "digitized_data": digitized_lookup,
            "validation_criteria": get_current_stage(state).get("validation_criteria"),
            "stage_id": state["current_stage_id"],
            "discrepancy_thresholds": DISCREPANCY_THRESHOLDS,
        }
    
    elif agent == "SupervisorAgent":
        return {
            "plan_summary": summarize_plan(state["plan"]),
            "progress": state["progress"],
            "validation_hierarchy": state["validation_hierarchy"],
            "figure_comparisons": state["figure_comparisons"],
            "current_stage_summary": summarize_current_stage(state),
            "runtime_budget": state["runtime_budget_remaining_seconds"],
            "revision_counts": {
                "design": state["design_revision_count"],
                "code": state["code_revision_count"],
                "analysis": state["analysis_revision_count"],
            },
            "backtrack_suggestion": state.get("backtrack_suggestion"),
            "systematic_discrepancies": state["systematic_discrepancies_identified"],
            # User feedback - critical for post-ask_user decisions
            "user_responses": state.get("user_responses", {}),
            "user_interactions": state.get("user_interactions", []),
            "pending_user_questions": state.get("pending_user_questions", []),
            "ask_user_trigger": state.get("ask_user_trigger"),
            "last_node_before_ask_user": state.get("last_node_before_ask_user"),
        }
    
    # ... other agents ...


def _compute_replan_reason(state: ReproState) -> str:
    """Summarize why replanning was triggered."""
    reasons = []
    if state.get("supervisor_feedback"):
        reasons.append(f"Supervisor: {state['supervisor_feedback']}")
    
    # Check recent user interactions
    recent_corrections = [
        ui for ui in state.get("user_interactions", [])
        if ui.get("interaction_type") in ["parameter_confirmation", "clarification"]
    ]
    if recent_corrections:
        reasons.append(f"User provided {len(recent_corrections)} correction(s)")
    
    return " | ".join(reasons) if reasons else "Replanning requested"


```

---

### Why This Design (Principle of Least Context)

**Benefits**:
1. **Prevents context bloat** - LLM context windows are limited
2. **Focused agents** - Each agent sees only what it needs to decide
3. **Faster inference** - Less context = faster response
4. **Easier debugging** - Each call is self-contained
5. **Cost control** - Fewer tokens = lower API costs
6. **Security** - Agents can't access irrelevant data

**Anti-patterns avoided**:
- ❌ Passing full `ReproState` to every agent
- ❌ Including full paper text in every call
- ❌ Sharing conversation history between agents
- ❌ Giving validators access to code details

### Why Runtime Injection?

**Benefits:**
- Constants defined once in code (`state.py`), no duplication
- Changes propagate automatically to all prompts
- State context is always current
- Prompts in `prompts/*.md` remain readable templates

**How it maintains single source of truth:**
```
schemas/state.py:DISCREPANCY_THRESHOLDS  ← canonical values
        ↓
src/prompts.py:format_thresholds_table() ← generates markdown
        ↓
Agent prompt at runtime                   ← LLM sees the values
```

## Prompt Adaptation Lifecycle

The PromptAdaptorAgent creates paper-specific modifications that customize agent behavior.
This section documents the complete lifecycle of these adaptations: creation, storage,
application, persistence, and restoration.

### Lifecycle Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ADAPTATION LIFECYCLE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. CREATION (adapt_prompts node)                                           │
│     │                                                                       │
│     ├─ PromptAdaptorAgent analyzes paper                                    │
│     ├─ Generates modifications with confidence levels                       │
│     ├─ Validates against forbidden modifications                            │
│     └─ Output: prompt_adaptations[] in state                                │
│                                                                             │
│  2. STORAGE                                                                 │
│     │                                                                       │
│     ├─ In-memory: state["prompt_adaptations"]                               │
│     ├─ On disk: outputs/<paper_id>/prompt_adaptations_<paper_id>.json       │
│     └─ Written: Immediately after adapt_prompts, updated at checkpoints     │
│                                                                             │
│  3. APPLICATION (every agent call)                                          │
│     │                                                                       │
│     ├─ src/prompts.py:build_prompt() reads state["prompt_adaptations"]      │
│     ├─ Filters adaptations for target agent                                 │
│     ├─ Injects adaptations into agent's base prompt                         │
│     └─ Result: Agent sees modified prompt                                   │
│                                                                             │
│  4. PERSISTENCE (at checkpoints)                                            │
│     │                                                                       │
│     ├─ Saved with full state in checkpoint JSON                             │
│     ├─ Also saved separately for human review                               │
│     └─ Includes: adaptations + domain_analysis + warnings                   │
│                                                                             │
│  5. RESTORATION (on resume)                                                 │
│     │                                                                       │
│     ├─ Loaded from checkpoint into state["prompt_adaptations"]              │
│     ├─ No re-execution of PromptAdaptorAgent needed                         │
│     └─ Agents continue with same adaptations                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
adapt_prompts                      plan, design, etc.
     │                                    │
     ▼                                    │
┌──────────────┐                          │
│PromptAdaptor │                          │
│    Agent     │                          │
└──────┬───────┘                          │
       │                                  │
       │ prompt_adaptations[]             │
       ▼                                  ▼
┌──────────────────────────────────────────────────┐
│                    ReproState                     │
│  ┌────────────────────────────────────────────┐  │
│  │ prompt_adaptations: [                      │  │
│  │   {id: "APPEND_001", target: "Designer",...}│  │
│  │   {id: "MOD_001", target: "Reviewer",...}  │  │
│  │ ]                                          │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
       │                                  │
       │                                  ▼
       │                         ┌──────────────┐
       │                         │ build_prompt │
       │                         │  (prompts.py)│
       │                         └──────┬───────┘
       │                                │
       │                                │ filtered adaptations
       │                                ▼
       │                         ┌──────────────┐
       │                         │ base_prompt  │
       │                         │     +        │
       │                         │ adaptations  │
       │                         │     =        │
       │                         │ final_prompt │
       │                         └──────────────┘
       │
       ▼ (at checkpoints)
┌──────────────────────────────────────────────────┐
│            Disk Artifacts                         │
│                                                   │
│  outputs/<paper_id>/                              │
│  ├── checkpoints/                                 │
│  │   └── checkpoint_*.json (includes adaptations) │
│  ├── prompt_adaptations_<paper_id>.json           │
│  └── adaptation_log.txt (human-readable)          │
└──────────────────────────────────────────────────┘
```

### When Adaptations Are Written to Disk

| Event | What's Saved | Location |
|-------|-------------|----------|
| After adapt_prompts | Full adaptation log | `_artifact_prompt_adaptations.json` |
| After plan | Checkpoint with adaptations | `checkpoints/checkpoint_after_plan_*.json` |
| Before ask_user | Full checkpoint | `checkpoints/checkpoint_before_ask_*.json` |
| After each stage | Updated checkpoint | `checkpoints/checkpoint_stage*_complete_*.json` |
| On completion | Final metrics (includes adaptation count) | `_artifact_metrics.json` |

### Adaptation Application (Implementation)

The `src/prompts.py` module applies adaptations at runtime:

```python
def build_prompt(agent_name: str, state: ReproState) -> str:
    """Build agent prompt with adaptations applied."""
    
    # 1. Load base prompt from file
    base_prompt = load_base_prompt(agent_name)
    
    # 2. Get adaptations for this agent
    adaptations = state.get("prompt_adaptations", [])
    agent_adaptations = [
        a for a in adaptations 
        if a["target_agent"] == agent_name
    ]
    
    # 3. Apply adaptations in order
    modified_prompt = base_prompt
    for adaptation in agent_adaptations:
        if adaptation["modification_type"] == "append":
            # Find target section and append
            modified_prompt = append_to_section(
                modified_prompt,
                adaptation["location"],
                adaptation["content"]
            )
        elif adaptation["modification_type"] == "modify":
            # Replace content
            modified_prompt = replace_content(
                modified_prompt,
                adaptation["original_content"],
                adaptation["content"]
            )
        elif adaptation["modification_type"] == "remove":
            # Comment out or mark as disabled
            modified_prompt = disable_content(
                modified_prompt,
                adaptation["content"]
            )
    
    # 4. Inject global rules (cannot be modified)
    final_prompt = inject_global_rules(modified_prompt)
    
    return final_prompt
```

### Checkpoint/Resume Behavior

**When saving a checkpoint:**
1. State including `prompt_adaptations` is serialized to JSON
2. Adaptations are preserved exactly as generated
3. No re-computation needed on resume

**When loading a checkpoint:**
1. State including `prompt_adaptations` is loaded
2. Graph resumes at the appropriate node
3. Subsequent agent calls use the loaded adaptations
4. PromptAdaptorAgent is NOT re-run

```python
# Resume from checkpoint
state = load_checkpoint("paper_123", checkpoint_name="after_plan")

# Adaptations are already in state - ready to use
assert "prompt_adaptations" in state
print(f"Loaded {len(state['prompt_adaptations'])} adaptations")

# Resume graph execution
result = app.invoke(state, resume_from="select_stage")
```

### Versioning and Base Prompt Changes

**Current Behavior (v1):**
- Adaptations reference sections by name (e.g., "Section B: MATERIALS")
- If base prompts change, adaptations may become invalid
- No automatic migration

**Handling Base Prompt Changes:**

| Scenario | Impact | Resolution |
|----------|--------|------------|
| Section renamed | Adaptation won't apply | Re-run PromptAdaptorAgent |
| Section removed | Adaptation becomes orphan | Re-run PromptAdaptorAgent |
| Content modified | Adaptation may conflict | Re-run PromptAdaptorAgent |
| New section added | No impact | Adaptations still apply |

**Best Practices:**
1. When updating base prompts, note which papers may be affected
2. Consider re-running adapt_prompts for active reproductions
3. Version base prompts alongside adaptation logs for reproducibility

**Future (v2):**
- Adaptation schema includes base prompt version hash
- Automatic detection of stale adaptations
- Migration tooling for prompt updates

### Reviewing Adaptations

**Human-readable log:**
```
outputs/<paper_id>/adaptation_log.txt

================================================================================
PROMPT ADAPTATIONS FOR: aluminum_nanoantenna_2013
Generated: 2025-12-01T10:30:00Z
Domain: plasmonics_strong_coupling
================================================================================

ADAPTATION #1: APPEND_001
  Target: SimulationDesignerAgent
  Type: append
  Confidence: 0.85
  Location: Section B: MATERIALS
  
  Content:
  ---
  For J-aggregate materials: Use Lorentzian oscillator model. Extract 
  parameters from absorption spectrum...
  ---
  
  Reason: Paper involves J-aggregate excitons which require specific 
  Lorentzian modeling not covered in base prompt.

ADAPTATION #2: MOD_001
  ...
```

**Programmatic access:**
```python
from schemas.state import load_checkpoint

state = load_checkpoint("paper_123")

# List all adaptations
for adaptation in state.get("prompt_adaptations", []):
    print(f"{adaptation['id']}: {adaptation['target_agent']}")
    print(f"  Type: {adaptation['modification_type']}")
    print(f"  Confidence: {adaptation['confidence']}")

# Filter by agent
designer_mods = [
    a for a in state["prompt_adaptations"]
    if a["target_agent"] == "SimulationDesignerAgent"
]
```

### Adaptation Metrics

Tracked in `state["metrics"]`:
- `prompt_adaptations_count`: Number of adaptations generated
- Per-adaptation effectiveness (filled post-reproduction)

Used by future PromptEvolutionAgent to learn which adaptations help.

### Troubleshooting Adaptations

| Issue | Symptom | Solution |
|-------|---------|----------|
| Adaptation not applied | Agent behaves like base prompt | Check `target_agent` matches exactly |
| Wrong section modified | Content appears in wrong place | Check `location` field |
| Adaptation conflicts | Agent confused or errors | Lower confidence threshold; choose one |
| Stale after base change | Section not found warnings | Re-run PromptAdaptorAgent |

**Debug mode:**
```python
# See exactly what prompt each agent receives
from src.prompts import build_prompt

final_prompt = build_prompt("SimulationDesignerAgent", state)
print(final_prompt)  # Shows base + adaptations applied
```

## Agent Data Flow

This section documents what data each agent receives and produces, clarifying the boundaries between agents.

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW BETWEEN AGENTS                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PromptAdaptorAgent                                                         │
│  ├─ IN:  paper_text_summary (abstract + Methods), paper_domain              │
│  └─ OUT: prompt_modifications[], adaptation_log, domain_analysis            │
│                    ↓                                                        │
│  PlannerAgent                                                               │
│  ├─ IN:  paper_text, paper_figures[], adapted_prompts                       │
│  └─ OUT: plan, assumptions, extracted_parameters[], progress (initialized)  │
│                    ↓                                                        │
│  SimulationDesignerAgent                                                    │
│  ├─ IN:  plan.stages[current], assumptions, extracted_parameters            │
│  │       reviewer_feedback (if revision)                                    │
│  └─ OUT: design_spec (geometry, materials, sources, BCs, monitors)          │
│          performance_estimate, new_assumptions                              │
│                    ↓                                                        │
│  CodeReviewerAgent (design review)                                          │
│  ├─ IN:  design_spec, plan.stages[current], assumptions                     │
│  └─ OUT: reviewer_verdict, reviewer_issues[], reviewer_feedback             │
│                    ↓ (if approved)                                          │
│  CodeGeneratorAgent                                                         │
│  ├─ IN:  design_spec, reviewer_feedback (if revision)                       │
│  └─ OUT: simulation_code, expected_outputs[]                                │
│                    ↓                                                        │
│  CodeReviewerAgent (code review)                                            │
│  ├─ IN:  simulation_code, design_spec, expected_outputs[]                   │
│  └─ OUT: reviewer_verdict, reviewer_issues[], reviewer_feedback             │
│                    ↓ (if approved)                                          │
│  run_code (not an agent - system execution)                                 │
│  ├─ IN:  simulation_code                                                    │
│  └─ OUT: stdout, stderr, output_files[], exit_code                          │
│                    ↓                                                        │
│  ExecutionValidatorAgent                                                    │
│  ├─ IN:  stdout, stderr, output_files[], expected_outputs[], exit_code      │
│  └─ OUT: execution_valid (bool), execution_issues[]                         │
│                    ↓ (if valid)                                             │
│  PhysicsSanityAgent                                                         │
│  ├─ IN:  output_files[] (data files), simulation_code                       │
│  └─ OUT: physics_valid (bool), physics_issues[], physics_validation         │
│                    ↓ (if valid)                                             │
│  ResultsAnalyzerAgent                                                       │
│  ├─ IN:  output_files[] (plots & data), paper_figures[target]               │
│  │       digitized_data (optional CSV), plan.validation_criteria            │
│  └─ OUT: figure_comparisons[], classification, discrepancies[]              │
│          analysis_summary                                                   │
│                    ↓                                                        │
│  ComparisonValidatorAgent                                                   │
│  ├─ IN:  figure_comparisons[], discrepancies[], analysis_summary            │
│  └─ OUT: comparison_valid (bool), comparison_issues[]                       │
│                    ↓ (if approved)                                          │
│  SupervisorAgent                                                            │
│  ├─ IN:  full state (plan, assumptions, progress, figure_comparisons,       │
│  │       validation_hierarchy, all verdicts)                                │
│  └─ OUT: supervisor_verdict, supervisor_feedback, progress updates          │
│          stage status updates, validation_hierarchy updates                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Shared vs Isolated Data

| Data Category | Scope | Agents with Access | Notes |
|---------------|-------|-------------------|-------|
| **paper_text** | Global (read-only) | All agents | Original paper content, never modified |
| **paper_figures** | Global (read-only) | Planner, ResultsAnalyzer, ComparisonValidator | Image paths for multimodal comparison |
| **plan** | Shared artifact | PlannerAgent (write), all others (read) | Persisted to `_artifact_plan.json` |
| **assumptions** | Shared artifact | Planner, Designer (write), all others (read) | Persisted to `_artifact_assumptions.json` |
| **progress** | Shared artifact | Supervisor (write), others (read) | Persisted to `_artifact_progress.json` |
| **design_spec** | Stage-local | Designer → Reviewer → Generator | Reset each stage |
| **simulation_code** | Stage-local | Generator → Reviewer → run_code | Reset each stage |
| **stage_outputs** | Stage-local | run_code → Validators → Analyzer | Reset each stage |
| **reviewer_feedback** | Loop-local | Reviewer → Designer/Generator | Used for revisions within a loop |
| **figure_comparisons** | Accumulated | Analyzer (write), Supervisor (read) | Persisted across stages |
| **validation_hierarchy** | Global (mutable) | Supervisor (write), select_stage (read) | Gates stage progression |

### Data Persistence Points

| Event | What Gets Saved | Location |
|-------|-----------------|----------|
| After plan | plan, assumptions, progress | `outputs/<paper_id>/_artifact_*.json` |
| After each supervisor approval | progress, figure_comparisons | `outputs/<paper_id>/_artifact_progress.json` |
| After stage completion | stage outputs | `outputs/<paper_id>/stage_*/` |
| Before ask_user | full checkpoint | `outputs/<paper_id>/checkpoints/` |
| After generate_report | final report | `outputs/<paper_id>/REPRODUCTION_REPORT_*.md` |

### Stage Completion Helpers

When a stage completes successfully, use these helper functions to update progress:

```python
from schemas.state import update_progress_stage_status, archive_stage_outputs_to_progress

# 1. Archive current stage_outputs to progress (preserves file references)
archive_stage_outputs_to_progress(state, stage_id, runtime_seconds=elapsed)

# 2. Update stage status
update_progress_stage_status(state, stage_id, "completed_success", 
    summary="Stage completed with excellent match to paper")
```

**`archive_stage_outputs_to_progress(state, stage_id, runtime_seconds=None)`**

Copies `state["stage_outputs"]["files"]` to `progress.stages[stage_id].outputs` before moving to the next stage. This is important because `stage_outputs` gets reset when entering a new stage.

- Automatically infers output type from file extension (spectrum_csv, field_plot, etc.)
- Records runtime_seconds if provided
- Should be called by supervisor_node when approving a stage as completed

**`update_progress_stage_status(state, stage_id, status, summary=None, invalidation_reason=None)`**

Updates the status of a stage in progress. Valid status values:
- `"not_started"`, `"in_progress"`, `"completed_success"`, `"completed_partial"`
- `"completed_failed"`, `"blocked"`, `"needs_rerun"`, `"invalidated"`
