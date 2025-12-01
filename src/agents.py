"""
Agent Node Implementations - Stubs for LangGraph Workflow

Each function is a LangGraph node that:
1. Receives the current ReproState
2. Performs agent-specific processing (TODO: implement with LLM calls)
3. Updates relevant state fields
4. Returns the updated state

These are currently stubs showing expected state mutations.
See prompts/*.md for the corresponding agent system prompts.
"""

from schemas.state import ReproState


def adapt_prompts_node(state: ReproState) -> ReproState:
    """PromptAdaptorAgent: Customize prompts for paper-specific needs."""
    state["workflow_phase"] = "adapting_prompts"
    # TODO: Implement prompt adaptation logic
    # - Analyze paper domain and techniques
    # - Generate prompt modifications
    # - Store in state["prompt_adaptations"]
    return state


def plan_node(state: ReproState) -> ReproState:
    """PlannerAgent: Analyze paper and create reproduction plan."""
    state["workflow_phase"] = "planning"
    # TODO: Implement planning logic
    # - Extract parameters from paper
    # - Classify figures
    # - Design staged reproduction plan
    # - Initialize assumptions
    return state


def plan_reviewer_node(state: ReproState) -> ReproState:
    """PlanReviewerAgent: Review reproduction plan before stage execution."""
    state["workflow_phase"] = "plan_review"
    # TODO: Implement plan review logic using prompts/plan_reviewer_agent.md
    # - Check coverage of reproducible figures
    # - Verify Stage 0 and Stage 1 present
    # - Validate parameter extraction
    # - Check assumptions quality
    # - Verify runtime estimates
    # - Set last_plan_review_verdict: "approve" | "needs_revision"
    return state


def select_stage_node(state: ReproState) -> dict:
    """
    Select next stage based on dependencies and validation hierarchy.
    
    Priority order:
    1. Stages with status "needs_rerun" (highest priority - backtrack targets)
    2. Stages with status "not_started" whose dependencies are satisfied
    3. No more stages to run (return None for current_stage_id)
    
    Skips stages with status:
    - "completed_success" / "completed_partial" / "completed_failed" - already done
    - "invalidated" - waiting for dependency to complete
    - "in_progress" - shouldn't happen, but skip
    - "blocked" - dependencies not met or budget exceeded
    
    Returns:
        Dict with state updates (LangGraph merges this into state)
    """
    from schemas.state import get_validation_hierarchy, STAGE_TYPE_TO_HIERARCHY_KEY
    
    progress = state.get("progress", {})
    stages = progress.get("stages", [])
    plan = state.get("plan", {})
    plan_stages = plan.get("stages", [])
    
    if not stages and not plan_stages:
        # No stages defined yet
        return {
            "workflow_phase": "stage_selection",
            "current_stage_id": None,
            "current_stage_type": None,
        }
    
    # Use plan stages if progress stages aren't initialized
    if not stages:
        stages = plan_stages
    
    # Get current validation hierarchy
    hierarchy = get_validation_hierarchy(state)
    
    # Priority 1: Find stages that need re-run (backtrack targets)
    for stage in stages:
        if stage.get("status") == "needs_rerun":
            return {
                "workflow_phase": "stage_selection",
                "current_stage_id": stage.get("stage_id"),
                "current_stage_type": stage.get("stage_type"),
                # Reset per-stage counters when entering a new stage
                "design_revision_count": 0,
                "code_revision_count": 0,
                "execution_failure_count": 0,
                "physics_failure_count": 0,
            }
    
    # Priority 2: Find not_started stages with satisfied dependencies
    for stage in stages:
        status = stage.get("status", "not_started")
        
        # Skip completed, in_progress, blocked, or invalidated stages
        if status in ["completed_success", "completed_partial", "completed_failed", 
                      "in_progress", "blocked", "invalidated"]:
            continue
        
        # Check if dependencies are satisfied
        dependencies = stage.get("dependencies", [])
        deps_satisfied = True
        
        for dep_id in dependencies:
            # Find the dependency stage
            dep_stage = next((s for s in stages if s.get("stage_id") == dep_id), None)
            if dep_stage:
                dep_status = dep_stage.get("status", "not_started")
                # Dependency must be completed (success or partial)
                if dep_status not in ["completed_success", "completed_partial"]:
                    deps_satisfied = False
                    break
        
        if not deps_satisfied:
            continue
        
        # Check validation hierarchy for stage type
        stage_type = stage.get("stage_type")
        
        # Enforce hierarchy: can't run later stages without earlier ones passing
        if stage_type == "SINGLE_STRUCTURE":
            if hierarchy.get("material_validation") not in ["passed", "partial"]:
                continue
        elif stage_type == "ARRAY_SYSTEM":
            if hierarchy.get("single_structure") not in ["passed", "partial"]:
                continue
        elif stage_type == "PARAMETER_SWEEP":
            # Parameter sweeps typically need at least single structure
            if hierarchy.get("single_structure") not in ["passed", "partial"]:
                continue
        
        # This stage is eligible
        return {
            "workflow_phase": "stage_selection",
            "current_stage_id": stage.get("stage_id"),
            "current_stage_type": stage_type,
            # Reset per-stage counters when entering a new stage
            "design_revision_count": 0,
            "code_revision_count": 0,
            "execution_failure_count": 0,
            "physics_failure_count": 0,
        }
    
    # No more stages to run
    return {
        "workflow_phase": "stage_selection",
        "current_stage_id": None,
        "current_stage_type": None,
    }


def simulation_designer_node(state: ReproState) -> ReproState:
    """SimulationDesignerAgent: Design simulation setup for current stage."""
    state["workflow_phase"] = "design"
    # TODO: Implement design logic
    # - Interpret geometry from plan
    # - Select materials
    # - Configure sources, BCs, monitors
    # - Estimate performance
    return state


def design_reviewer_node(state: ReproState) -> ReproState:
    """DesignReviewerAgent: Review simulation design before code generation."""
    state["workflow_phase"] = "design_review"
    # TODO: Implement design review logic using prompts/design_reviewer_agent.md
    # - Check geometry matches paper
    # - Verify physics setup is correct
    # - Validate material choices
    # - Check unit system (a_unit)
    # - Verify source/excitation setup
    # - Set last_design_review_verdict: "approve" | "needs_revision"
    return state


def code_reviewer_node(state: ReproState) -> ReproState:
    """CodeReviewerAgent: Review generated code before execution."""
    state["workflow_phase"] = "code_review"
    # TODO: Implement code review logic using prompts/code_reviewer_agent.md
    # - Verify a_unit matches design
    # - Check Meep API usage
    # - Validate numerics implementation
    # - Check code quality (no plt.show, etc.)
    # - Set last_code_review_verdict: "approve" | "needs_revision"
    return state


def code_generator_node(state: ReproState) -> ReproState:
    """CodeGeneratorAgent: Generate Python+Meep code from approved design."""
    state["workflow_phase"] = "code_generation"
    # TODO: Implement code generation logic
    # - Convert design to Meep code
    # - Include progress prints
    # - Set expected outputs
    return state


def execution_validator_node(state: ReproState) -> ReproState:
    """ExecutionValidatorAgent: Validate simulation ran correctly."""
    state["workflow_phase"] = "execution_validation"
    # TODO: Implement execution validation logic
    # - Check completion status
    # - Verify output files exist
    # - Check for NaN/Inf in data
    return state


def physics_sanity_node(state: ReproState) -> ReproState:
    """PhysicsSanityAgent: Validate physics of results."""
    state["workflow_phase"] = "physics_validation"
    # TODO: Implement physics validation logic
    # - Check conservation laws (T + R + A ≈ 1)
    # - Verify value ranges
    # - Check numerical quality
    return state


def results_analyzer_node(state: ReproState) -> ReproState:
    """ResultsAnalyzerAgent: Compare results to paper figures."""
    state["workflow_phase"] = "analysis"
    # TODO: Implement analysis logic
    # - Compare simulation outputs to paper figures
    # - Compute discrepancies
    # - Classify reproduction quality
    return state


def comparison_validator_node(state: ReproState) -> ReproState:
    """ComparisonValidatorAgent: Validate comparison accuracy."""
    state["workflow_phase"] = "comparison_validation"
    # TODO: Implement comparison validation logic
    # - Verify math is correct
    # - Check classifications match numbers
    # - Validate discrepancy documentation
    return state


def supervisor_node(state: ReproState) -> ReproState:
    """SupervisorAgent: Big-picture assessment and decisions."""
    state["workflow_phase"] = "supervision"
    # TODO: Implement supervision logic
    # - Assess overall progress
    # - Check validation hierarchy
    # - Decide: continue, replan, ask_user, backtrack
    return state


def ask_user_node(state: ReproState) -> ReproState:
    """Pause for user input."""
    state["workflow_phase"] = "awaiting_user"
    state["awaiting_user_input"] = True
    # TODO: Implement user interaction logic
    # - Present pending questions
    # - Wait for responses
    # - Log interaction
    return state


def material_checkpoint_node(state: ReproState) -> dict:
    """
    Mandatory material validation checkpoint after Stage 0.
    
    This node ALWAYS routes to ask_user to require user confirmation
    of material validation results before proceeding to Stage 1+.
    
    Also extracts and populates `validated_materials` from the plan,
    which Code Generator will use to find material file paths.
    
    Per global_rules.md RULE 0A:
    "After Stage 0 completes, you MUST pause and ask the user to confirm
    the material optical constants are correct before proceeding."
    
    Returns:
        Dict with state updates including pending_user_questions and validated_materials
    """
    from schemas.state import get_validation_hierarchy
    
    # Get material validation results from progress
    progress = state.get("progress", {})
    stages = progress.get("stages", [])
    
    # Find Stage 0 (material validation) results
    stage0_info = None
    for stage in stages:
        if stage.get("stage_type") == "MATERIAL_VALIDATION":
            stage0_info = stage
            break
    
    # Get output files from stage_outputs
    stage_outputs = state.get("stage_outputs", {})
    output_files = stage_outputs.get("files", [])
    plot_files = [f for f in output_files if f.endswith(('.png', '.pdf', '.jpg'))]
    
    # Extract validated materials from plan parameters
    validated_materials = _extract_validated_materials(state)
    
    # Build the checkpoint question per global_rules.md RULE 0A format
    question = _format_material_checkpoint_question(state, stage0_info, plot_files, validated_materials)
    
    return {
        "workflow_phase": "material_checkpoint",
        "pending_user_questions": [question],
        "awaiting_user_input": True,
        "ask_user_trigger": "material_checkpoint",
        "last_node_before_ask_user": "material_checkpoint",
        "validated_materials": validated_materials,
    }


def _extract_validated_materials(state: ReproState) -> list:
    """
    Extract material information from plan and build validated_materials list.
    
    Scans extracted_parameters for material-related entries and maps them
    to file paths in the materials/ directory.
    
    Returns:
        List of dicts: [{"material": "gold", "source": "palik", "path": "materials/palik_gold.csv"}]
    """
    import os
    
    plan = state.get("plan", {})
    extracted_params = plan.get("extracted_parameters", [])
    assumptions = state.get("assumptions", {})
    
    validated_materials = []
    seen_materials = set()
    
    # Known material databases and their naming conventions
    KNOWN_SOURCES = ["palik", "johnson_christy", "rakic", "malitson"]
    KNOWN_MATERIALS = ["gold", "silver", "aluminum", "silicon", "sio2", "glass"]
    
    # Scan extracted parameters for material info
    for param in extracted_params:
        name = param.get("name", "").lower()
        value = str(param.get("value", "")).lower()
        
        # Look for material-related parameters
        if "material" in name or any(mat in name for mat in KNOWN_MATERIALS):
            material = None
            source = "palik"  # Default source
            
            # Try to identify the material
            for mat in KNOWN_MATERIALS:
                if mat in value or mat in name:
                    material = mat
                    break
            
            # Try to identify the source
            for src in KNOWN_SOURCES:
                if src in value.lower():
                    source = src
                    break
            
            if material and material not in seen_materials:
                # Construct file path
                path = f"materials/{source}_{material}.csv"
                
                validated_materials.append({
                    "material": material,
                    "source": source,
                    "path": path,
                    "from_parameter": param.get("name"),
                })
                seen_materials.add(material)
    
    # Also check assumptions for material choices
    global_assumptions = assumptions.get("global_assumptions", {})
    material_assumptions = global_assumptions.get("materials", [])
    
    for assumption in material_assumptions:
        if isinstance(assumption, dict):
            desc = assumption.get("description", "").lower()
            for mat in KNOWN_MATERIALS:
                if mat in desc and mat not in seen_materials:
                    source = "palik"
                    for src in KNOWN_SOURCES:
                        if src in desc:
                            source = src
                            break
                    
                    validated_materials.append({
                        "material": mat,
                        "source": source,
                        "path": f"materials/{source}_{mat}.csv",
                        "from_assumption": assumption.get("description"),
                    })
                    seen_materials.add(mat)
    
    return validated_materials


def _format_material_checkpoint_question(
    state: ReproState, 
    stage0_info: dict, 
    plot_files: list,
    validated_materials: list
) -> str:
    """Format the material checkpoint question per global_rules.md RULE 0A."""
    paper_id = state.get("paper_id", "unknown")
    
    # Format validated materials
    if validated_materials:
        materials_info = []
        for mat in validated_materials:
            materials_info.append(
                f"- {mat['material'].upper()}: source={mat['source']}, file={mat['path']}"
            )
    else:
        materials_info = ["- No materials automatically detected"]
    
    # Format plot files list
    plots_text = "\n".join(f"- {f}" for f in plot_files) if plot_files else "- No plots generated"
    
    question = f"""
═══════════════════════════════════════════════════════════════════════
MANDATORY MATERIAL VALIDATION CHECKPOINT
═══════════════════════════════════════════════════════════════════════

Stage 0 (Material Validation) has completed for paper: {paper_id}

**Validated materials (will be used for all subsequent stages):**
{chr(10).join(materials_info)}

**Generated plots:**
{plots_text}

Please review the material optical constants comparison plots above.

**Required confirmation:**

Do the simulated optical constants (n, k, ε) match the paper's data 
within acceptable tolerance?

Options:
1. APPROVE - Material validation looks correct, proceed to Stage 1
2. CHANGE_DATABASE - Use different material database (specify which)
3. CHANGE_MATERIAL - Paper uses different material than assumed (specify which)
4. NEED_HELP - Unclear how to validate, need guidance

Note: If you APPROVE, the validated_materials list above will be passed
to Code Generator for all subsequent stages.

Please respond with your choice and any notes.
═══════════════════════════════════════════════════════════════════════
"""
    return question


def generate_report_node(state: ReproState) -> ReproState:
    """Generate final reproduction report."""
    state["workflow_phase"] = "reporting"
    # TODO: Implement report generation logic
    # - Compile figure comparisons
    # - Document assumptions
    # - Generate REPRODUCTION_REPORT.md
    return state


def handle_backtrack_node(state: ReproState) -> dict:
    """
    Process cross-stage backtracking.
    
    When SupervisorAgent decides to backtrack (verdict="backtrack_to_stage"),
    this node:
    1. Reads the backtrack_decision from state
    2. Marks the target stage as "needs_rerun"
    3. Marks all dependent stages as "invalidated"
    4. Increments the backtrack_count
    5. Clears working data to prepare for re-run
    
    Returns:
        Dict with state updates (LangGraph merges this into state)
    """
    import copy
    
    decision = state.get("backtrack_decision", {})
    if not decision or not decision.get("accepted"):
        # No valid backtrack decision - shouldn't happen, but handle gracefully
        return {"workflow_phase": "backtracking"}
    
    target_id = decision.get("target_stage_id", "")
    stages_to_invalidate = decision.get("stages_to_invalidate", [])
    
    # Deep copy progress to avoid mutating original
    progress = copy.deepcopy(state.get("progress", {}))
    stages = progress.get("stages", [])
    
    # Update stage statuses
    for stage in stages:
        stage_id = stage.get("stage_id", "")
        if stage_id == target_id:
            # Target stage: mark for re-run
            stage["status"] = "needs_rerun"
        elif stage_id in stages_to_invalidate:
            # Dependent stages: mark as invalidated
            stage["status"] = "invalidated"
    
    # Build return dict with all state updates
    return {
        "workflow_phase": "backtracking",
        "progress": progress,
        "current_stage_id": target_id,
        "backtrack_count": state.get("backtrack_count", 0) + 1,
        # Clear working data to prepare for re-run
        "code": None,
        "design_description": None,
        "stage_outputs": {},
        "run_error": None,
        "analysis_summary": None,
        # Track invalidated stages
        "invalidated_stages": stages_to_invalidate,
        # Clear verdicts
        "last_design_review_verdict": None,
        "last_code_review_verdict": None,
        "supervisor_verdict": None,
        # Reset per-stage counters
        "design_revision_count": 0,
        "code_revision_count": 0,
        "execution_failure_count": 0,
        "physics_failure_count": 0,
    }

