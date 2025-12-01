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

import os
import signal
from typing import Dict, Any, Optional

from schemas.state import ReproState, save_checkpoint, check_context_before_node


# ═══════════════════════════════════════════════════════════════════════
# Context Window Management
# ═══════════════════════════════════════════════════════════════════════

def _check_context_or_escalate(state: ReproState, node_name: str) -> Optional[Dict[str, Any]]:
    """
    Check context before LLM call. Returns state updates if escalation needed, None otherwise.
    
    This is called explicitly at the start of each agent node that makes LLM calls.
    If context is critical and cannot be auto-recovered, prepares escalation to user.
    
    Args:
        state: Current ReproState
        node_name: Name of the node about to make LLM call
        
    Returns:
        None if safe to proceed (or with minor auto-recovery applied)
        Dict with state updates if escalation to user is needed
    """
    check = check_context_before_node(state, node_name, auto_recover=True)
    
    if check["ok"]:
        # Safe to proceed, possibly with state updates from auto-recovery
        return check.get("state_updates") if check.get("state_updates") else None
    
    if check["escalate"]:
        # Must ask user - return state updates to trigger ask_user
        return {
            "pending_user_questions": [check["user_question"]],
            "awaiting_user_input": True,
            "ask_user_trigger": "context_overflow",
            "last_node_before_ask_user": node_name,
        }
    
    # Shouldn't reach here, but fallback to escalation
    return {
        "pending_user_questions": [f"Context overflow in {node_name}. How should we proceed?"],
        "awaiting_user_input": True,
        "ask_user_trigger": "context_overflow",
        "last_node_before_ask_user": node_name,
    }


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


def ask_user_node(state: ReproState) -> Dict[str, Any]:
    """
    CLI-based user interaction node.
    
    Prompts user in terminal for input. If user doesn't respond within timeout
    (Ctrl+C or timeout), saves checkpoint and exits gracefully.
    
    Environment variables:
        REPROLAB_USER_TIMEOUT_SECONDS: Override default timeout (default: 86400 = 24h)
        REPROLAB_NON_INTERACTIVE: If "1", immediately save checkpoint and exit
        
    Returns:
        Dict with state updates (user_responses, cleared pending questions)
    """
    timeout_seconds = int(os.environ.get("REPROLAB_USER_TIMEOUT_SECONDS", "86400"))
    non_interactive = os.environ.get("REPROLAB_NON_INTERACTIVE", "0") == "1"
    
    questions = state.get("pending_user_questions", [])
    trigger = state.get("ask_user_trigger", "unknown")
    paper_id = state.get("paper_id", "unknown")
    
    if not questions:
        return {
            "awaiting_user_input": False,
            "workflow_phase": "awaiting_user",
        }
    
    # Non-interactive mode: save and exit immediately
    if non_interactive:
        print("\n" + "=" * 60)
        print("USER INPUT REQUIRED (non-interactive mode)")
        print("=" * 60)
        print(f"\nPaper: {paper_id}")
        print(f"Trigger: {trigger}")
        for i, q in enumerate(questions, 1):
            print(f"\nQuestion {i}:\n{q}")
        
        checkpoint_path = save_checkpoint(state, f"awaiting_user_{trigger}")
        print(f"\n✓ Checkpoint saved: {checkpoint_path}")
        print("\nResume with:")
        print(f"  python -m src.graph --resume {checkpoint_path}")
        raise SystemExit(0)
    
    # Interactive mode: prompt user
    print("\n" + "=" * 60)
    print("USER INPUT REQUIRED")
    print("=" * 60)
    print(f"Paper: {paper_id}")
    print(f"Trigger: {trigger}")
    print("(Press Ctrl+C to save checkpoint and exit)")
    print("=" * 60)
    
    responses = {}
    
    def timeout_handler(signum, frame):
        raise TimeoutError("User response timeout")
    
    try:
        # Set timeout (Unix only - SIGALRM not available on Windows)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- Question {i}/{len(questions)} ---")
            print(f"\n{question}")
            print("-" * 40)
            
            # Multi-line input support: empty line ends input
            print("(Enter your response, then press Enter twice to submit)")
            lines = []
            while True:
                try:
                    line = input()
                    if line == "" and lines:
                        break
                    lines.append(line)
                except EOFError:
                    break
            
            response = "\n".join(lines).strip()
            if not response:
                response = input("Your response (single line): ").strip()
            
            responses[question] = response
            print(f"✓ Response recorded")
        
        # Cancel timeout
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        
        print("\n" + "=" * 60)
        print("✓ All responses collected")
        print("=" * 60)
            
    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted by user (Ctrl+C)")
        checkpoint_path = save_checkpoint(state, f"interrupted_{trigger}")
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        print("\nResume later with:")
        print(f"  python -m src.graph --resume {checkpoint_path}")
        raise SystemExit(0)
        
    except TimeoutError:
        print(f"\n\n⚠ User response timeout ({timeout_seconds}s)")
        checkpoint_path = save_checkpoint(state, f"timeout_{trigger}")
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        print("\nResume later with:")
        print(f"  python -m src.graph --resume {checkpoint_path}")
        raise SystemExit(0)
        
    except EOFError:
        print(f"\n\n⚠ End of input (EOF)")
        checkpoint_path = save_checkpoint(state, f"eof_{trigger}")
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        print("\nResume later with:")
        print(f"  python -m src.graph --resume {checkpoint_path}")
        raise SystemExit(0)
    
    return {
        "user_responses": {**state.get("user_responses", {}), **responses},
        "pending_user_questions": [],
        "awaiting_user_input": False,
        "workflow_phase": "awaiting_user",
    }


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
    
    Uses materials/index.json as the authoritative source for material data.
    Scans extracted_parameters for material-related entries and matches them
    against material_id values in the index.
    
    Returns:
        List of dicts with material_id, data_file, drude_lorentz_fit, etc.
    """
    import json
    import os
    
    plan = state.get("plan", {})
    extracted_params = plan.get("extracted_parameters", [])
    assumptions = state.get("assumptions", {})
    
    # Load material database
    material_db = _load_material_database()
    if not material_db:
        return []
    
    # Build lookup tables from material database
    material_lookup = {}
    for mat in material_db.get("materials", []):
        mat_id = mat.get("material_id", "")
        material_lookup[mat_id] = mat
        
        # Also index by simple name (e.g., "silver" -> "palik_silver")
        # Extract simple name from material_id (e.g., "palik_silver" -> "silver")
        parts = mat_id.split("_")
        if len(parts) >= 2:
            simple_name = parts[-1]  # e.g., "silver", "gold", "aluminum"
            if simple_name not in material_lookup:
                material_lookup[simple_name] = mat
    
    validated_materials = []
    seen_material_ids = set()
    
    # Scan extracted parameters for material info
    for param in extracted_params:
        name = param.get("name", "").lower()
        value = str(param.get("value", "")).lower()
        
        # Look for material-related parameters
        if "material" in name:
            matched_material = _match_material_from_text(value, material_lookup)
            if matched_material and matched_material["material_id"] not in seen_material_ids:
                validated_materials.append(_format_validated_material(
                    matched_material, 
                    from_source=f"parameter: {param.get('name')}"
                ))
                seen_material_ids.add(matched_material["material_id"])
    
    # Also check assumptions for material choices
    global_assumptions = assumptions.get("global_assumptions", {})
    material_assumptions = global_assumptions.get("materials", [])
    
    for assumption in material_assumptions:
        if isinstance(assumption, dict):
            desc = assumption.get("description", "").lower()
            matched_material = _match_material_from_text(desc, material_lookup)
            if matched_material and matched_material["material_id"] not in seen_material_ids:
                validated_materials.append(_format_validated_material(
                    matched_material,
                    from_source=f"assumption: {assumption.get('description', '')[:50]}"
                ))
                seen_material_ids.add(matched_material["material_id"])
    
    return validated_materials


def _load_material_database() -> dict:
    """Load materials/index.json database."""
    import json
    import os
    
    index_path = os.path.join(os.path.dirname(__file__), "..", "materials", "index.json")
    if not os.path.exists(index_path):
        # Try relative to current working directory
        index_path = "materials/index.json"
    
    try:
        with open(index_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load materials/index.json: {e}")
        return {}


def _match_material_from_text(text: str, material_lookup: dict) -> dict:
    """
    Match material name from text against material database.
    
    Returns the best matching material entry or None.
    """
    text_lower = text.lower()
    
    # Priority 1: Exact material_id match (e.g., "palik_gold")
    for mat_id, mat_entry in material_lookup.items():
        if mat_id in text_lower:
            return mat_entry
    
    # Priority 2: Simple material name match (e.g., "gold", "silver")
    simple_names = ["gold", "silver", "aluminum", "silicon", "sio2", "glass", "water", "air"]
    for name in simple_names:
        if name in text_lower:
            # Find the best match in lookup (prefer entries with csv_available=true)
            candidates = [v for k, v in material_lookup.items() if name in k]
            csv_available = [c for c in candidates if c.get("csv_available", False)]
            if csv_available:
                return csv_available[0]
            elif candidates:
                return candidates[0]
    
    return None


def _format_validated_material(mat_entry: dict, from_source: str) -> dict:
    """Format a material database entry for validated_materials list."""
    data_file = mat_entry.get("data_file")
    path = f"materials/{data_file}" if data_file else None
    
    return {
        "material_id": mat_entry.get("material_id"),
        "name": mat_entry.get("name"),
        "source": mat_entry.get("source"),
        "path": path,
        "csv_available": mat_entry.get("csv_available", False),
        "drude_lorentz_fit": mat_entry.get("drude_lorentz_fit"),
        "wavelength_range_nm": mat_entry.get("wavelength_range_nm"),
        "from": from_source,
    }


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

