"""
LangGraph State Definition for Paper Reproduction System

This module defines the TypedDict state that flows through the LangGraph
state machine. State is persisted between nodes and can be checkpointed.

═══════════════════════════════════════════════════════════════════════════════
IMPORTANT: JSON SCHEMAS ARE THE SOURCE OF TRUTH
═══════════════════════════════════════════════════════════════════════════════

Many types used in this system are defined in JSON Schema files (schemas/*.json).
These JSON schemas are the canonical definitions and should be used to generate
Python TypedDicts to ensure consistency.

JSON Schema Files:
- plan_schema.json        → Plan structure, ExtractedParameter, Stage, Target
- assumptions_schema.json → Assumption, GeometryInterpretation
- progress_schema.json    → StageProgress, Output, Discrepancy, UserInteraction
- metrics_schema.json     → AgentCallMetric, StageMetric, MetricsLog
- report_schema.json      → FigureComparison, OverallAssessment, Conclusions
- prompt_adaptations_schema.json → Adaptation records

To generate Python types from JSON schemas:

    pip install datamodel-code-generator
    
    datamodel-codegen \\
        --input schemas/plan_schema.json \\
        --input schemas/progress_schema.json \\
        --input schemas/metrics_schema.json \\
        --input schemas/report_schema.json \\
        --input-file-type jsonschema \\
        --output-model-type typing.TypedDict \\
        --output schemas/generated_types.py

Then import generated types:
    from schemas.generated_types import ExtractedParameter, Discrepancy, ...

This file (state.py) contains:
1. Workflow-specific types NOT in JSON schemas (ReproState, RuntimeConfig, etc.)
2. Constants and thresholds
3. Helper functions for state management
4. Validation hierarchy mappings

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations  # Enable forward references for type hints

from typing import TypedDict, Optional, List, Dict, Any
from typing_extensions import NotRequired
from datetime import datetime, timezone
import re


# ═══════════════════════════════════════════════════════════════════════
# Workflow-Specific Type Definitions
# ═══════════════════════════════════════════════════════════════════════
#
# These types are specific to the LangGraph workflow and are NOT defined
# in the JSON schemas. They represent internal state that doesn't get
# persisted to JSON files.
#
# For types that ARE defined in JSON schemas (ExtractedParameter,
# Discrepancy, StageProgress, FigureComparison, etc.), generate them
# from the schemas using datamodel-code-generator.
# ═══════════════════════════════════════════════════════════════════════


class ReviewerIssue(TypedDict):
    """
    An issue identified by CodeReviewerAgent or validation agents.
    
    This is workflow-internal and not persisted to JSON schemas.
    """
    severity: str  # blocking | major | minor
    category: str  # geometry | material | source | numerical | analysis | documentation
    description: str
    suggested_fix: str
    reference: NotRequired[str]


class ValidationHierarchyStatus(TypedDict):
    """Status of validation hierarchy stages.
    
    Values: "passed" | "partial" | "failed" | "not_done"
    
    IMPORTANT: These values differ from StageProgress.status values!
    See STAGE_STATUS_TO_HIERARCHY_MAPPING below for the mapping.
    """
    material_validation: str  # passed | failed | partial | not_done
    single_structure: str
    arrays_systems: str
    parameter_sweeps: str


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION HIERARCHY - SINGLE SOURCE OF TRUTH
# ═══════════════════════════════════════════════════════════════════════════════
#
# The validation hierarchy is COMPUTED from progress["stages"] on demand.
# This eliminates the need for manual synchronization between two state systems.
#
# Stage Status → Hierarchy Value Mapping:
#
# ┌─────────────────────────┬──────────────────────────┬─────────────────────────┐
# │ Stage Status            │ Validation Hierarchy     │ Meaning                 │
# │ (StageProgress.status)  │ (computed value)         │                         │
# ├─────────────────────────┼──────────────────────────┼─────────────────────────┤
# │ "completed_success"     │ "passed"                 │ Stage passed all checks │
# │ "completed_partial"     │ "partial"                │ Acceptable gaps exist   │
# │ "completed_failed"      │ "failed"                 │ Stage failed validation │
# │ "not_started"           │ "not_done"               │ Stage hasn't run yet    │
# │ "in_progress"           │ "not_done"               │ Currently running       │
# │ "blocked"               │ "not_done"               │ Skipped (budget/deps)   │
# │ "needs_rerun"           │ "not_done"               │ Backtrack target        │
# │ "invalidated"           │ "not_done"               │ Will re-run when ready  │
# └─────────────────────────┴──────────────────────────┴─────────────────────────┘
#
# STAGE TYPE → VALIDATION HIERARCHY KEY MAPPING:
#
# ┌─────────────────────────┬──────────────────────────┐
# │ Stage Type (plan)       │ Validation Hierarchy Key │
# ├─────────────────────────┼──────────────────────────┤
# │ "MATERIAL_VALIDATION"   │ "material_validation"    │
# │ "SINGLE_STRUCTURE"      │ "single_structure"       │
# │ "ARRAY_SYSTEM"          │ "arrays_systems"         │
# │ "PARAMETER_SWEEP"       │ "parameter_sweeps"       │
# │ "COMPLEX_PHYSICS"       │ (not in hierarchy)       │
# └─────────────────────────┴──────────────────────────┘
#
# ═══════════════════════════════════════════════════════════════════════════════

# Mapping from stage status to validation hierarchy value
STAGE_STATUS_TO_HIERARCHY_MAPPING = {
    "completed_success": "passed",
    "completed_partial": "partial",
    "completed_failed": "failed",
    "not_started": "not_done",
    "in_progress": "not_done",
    "blocked": "not_done",
    "needs_rerun": "not_done",
    "invalidated": "not_done",
}

# Mapping from stage type to validation hierarchy key
STAGE_TYPE_TO_HIERARCHY_KEY = {
    "MATERIAL_VALIDATION": "material_validation",
    "SINGLE_STRUCTURE": "single_structure",
    "ARRAY_SYSTEM": "arrays_systems",
    "PARAMETER_SWEEP": "parameter_sweeps",
    # "COMPLEX_PHYSICS" doesn't have a hierarchy key (optional stage type)
}


def get_validation_hierarchy(state: dict) -> "ValidationHierarchyStatus":
    """
    Compute validation hierarchy from progress stages (single source of truth).
    
    This function derives the validation hierarchy on demand from the 
    progress["stages"] data. There is no need to manually sync state—
    the hierarchy is always computed fresh from the canonical source.
    
    Args:
        state: ReproState dict containing progress with stages
        
    Returns:
        ValidationHierarchyStatus dict with current hierarchy state
        
    Example:
        # In routing logic or agent code:
        hierarchy = get_validation_hierarchy(state)
        if hierarchy["material_validation"] != "passed":
            # Cannot proceed to Stage 1
            pass
            
        # In select_stage logic:
        hierarchy = get_validation_hierarchy(state)
        if hierarchy["single_structure"] in ["passed", "partial"]:
            # Can proceed to array/sweep stages
            pass
    """
    # Default hierarchy - everything starts as not_done
    hierarchy: ValidationHierarchyStatus = {
        "material_validation": "not_done",
        "single_structure": "not_done",
        "arrays_systems": "not_done",
        "parameter_sweeps": "not_done",
    }
    
    # Get stages from progress (single source of truth)
    progress = state.get("progress", {})
    stages = progress.get("stages", [])
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONSISTENCY CHECK: Validate plan ↔ progress alignment
    # ═══════════════════════════════════════════════════════════════════════
    plan = state.get("plan")
    if plan is None or not isinstance(plan, dict):
        plan = {}
    plan_stages = plan.get("stages", [])
    
    if plan_stages and not stages:
        # Plan exists but progress not initialized - this is an error state
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "Plan has stages but progress is not initialized. "
            "This indicates a bug in plan → progress initialization."
        )
        # Return failed hierarchy to prevent proceeding
        return {
            "material_validation": "failed",
            "single_structure": "failed",
            "arrays_systems": "failed",
            "parameter_sweeps": "failed",
        }
    
    # Check that all plan stage types have corresponding progress entries
    if plan_stages and stages:
        plan_stage_ids = {s.get("stage_id") for s in plan_stages}
        progress_stage_ids = {s.get("stage_id") for s in stages}
        missing_stages = plan_stage_ids - progress_stage_ids
        
        if missing_stages:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Plan has {len(missing_stages)} stages missing from progress: {missing_stages}. "
                "This indicates progress initialization failure."
            )
            # Mark affected hierarchy levels as failed
            for stage_id in missing_stages:
                plan_stage = next((s for s in plan_stages if s.get("stage_id") == stage_id), None)
                if plan_stage:
                    stage_type = plan_stage.get("stage_type")
                    hierarchy_key = STAGE_TYPE_TO_HIERARCHY_KEY.get(stage_type)
                    if hierarchy_key and hierarchy_key in hierarchy:
                        hierarchy[hierarchy_key] = "failed"
    
    if not stages:
        return hierarchy
    
    # Compute hierarchy from stage statuses
    # Aggregation logic:
    # - If ANY stage of a type is "failed", "blocked", "invalidated", or "needs_rerun" -> type is "failed" (blocking)
    # - If ALL stages of a type are "completed_success" -> type is "passed"
    # - If stages are mixed "completed_success"/"completed_partial" -> type is "partial"
    # - If ANY stage is "not_started"/"in_progress" -> type is "not_done"
    
    # First, collect all statuses for each hierarchy key
    status_map = {k: [] for k in hierarchy.keys()}
    
    for stage in stages:
        stage_type = stage.get("stage_type")
        stage_status = stage.get("status", "not_started")
        
        hierarchy_key = STAGE_TYPE_TO_HIERARCHY_KEY.get(stage_type)
        if hierarchy_key and hierarchy_key in status_map:
            status_map[hierarchy_key].append(stage_status)
            
    # Now compute aggregate status for each key
    for key, statuses in status_map.items():
        # If hierarchy was already set to "failed" due to missing stages, keep it
        if hierarchy.get(key) == "failed":
            continue
        if not statuses:
            hierarchy[key] = "not_done" # No stages of this type exist/planned yet
            continue
            
        if any(s in ["completed_failed", "blocked", "invalidated", "needs_rerun"] for s in statuses):
            hierarchy[key] = "failed"
        elif any(s in ["not_started", "in_progress"] for s in statuses):
            hierarchy[key] = "not_done"
        elif all(s == "completed_success" for s in statuses):
            hierarchy[key] = "passed"
        else:
            # Mixed success/partial implies partial
            hierarchy[key] = "partial"
    
    return hierarchy


def get_plan_stage_types(state: dict) -> set:
    """
    Get the set of stage types present in this paper's plan.
    
    This enables paper-dependent validation hierarchy. Not all papers have
    all stage types - some skip single structure, some have no parameter sweeps.
    
    Args:
        state: ReproState dict containing plan with stages
        
    Returns:
        Set of stage type strings present in the plan
        
    Example:
        >>> stage_types = get_plan_stage_types(state)
        >>> if "SINGLE_STRUCTURE" not in stage_types:
        ...     # This paper skips single structure (e.g., photonic crystal)
        ...     # ARRAY_SYSTEM can proceed without single_structure passing
    """
    plan = state.get("plan", {})
    stages = plan.get("stages", [])
    return {s.get("stage_type") for s in stages if s.get("stage_type")}


def check_hierarchy_gate(
    state: dict,
    target_stage_type: str
) -> tuple:
    """
    Check if a stage type can proceed based on paper-dependent validation hierarchy.
    
    This function implements paper-adaptive hierarchy checks. It only enforces
    gates for stage types that actually exist in the plan.
    
    Args:
        state: ReproState dict
        target_stage_type: Stage type to check (e.g., "ARRAY_SYSTEM")
        
    Returns:
        Tuple of (can_proceed: bool, blocking_reason: str or None)
        
    Example:
        >>> can_proceed, reason = check_hierarchy_gate(state, "ARRAY_SYSTEM")
        >>> if not can_proceed:
        ...     print(f"Blocked: {reason}")
    """
    hierarchy = get_validation_hierarchy(state)
    plan_stage_types = get_plan_stage_types(state)
    
    # Material validation is ALWAYS required
    if hierarchy["material_validation"] != "passed":
        return (False, "Material validation (Stage 0) must pass first")
    
    if target_stage_type == "ARRAY_SYSTEM":
        # Only require single_structure if it exists in this plan
        if "SINGLE_STRUCTURE" in plan_stage_types:
            if hierarchy["single_structure"] != "passed":
                return (False, "Single structure validation must pass first")
        # If no SINGLE_STRUCTURE stage exists, ARRAY_SYSTEM can proceed
        return (True, None)
    
    if target_stage_type == "PARAMETER_SWEEP":
        # Sweeps can follow either single structure OR array, depending on plan
        if "ARRAY_SYSTEM" in plan_stage_types:
            if hierarchy["arrays_systems"] != "passed":
                return (False, "Array/system validation must pass first")
        elif "SINGLE_STRUCTURE" in plan_stage_types:
            if hierarchy["single_structure"] != "passed":
                return (False, "Single structure validation must pass first")
        return (True, None)
    
    if target_stage_type == "COMPLEX_PHYSICS":
        # Complex physics requires ALL applicable prerequisite stages
        if "PARAMETER_SWEEP" in plan_stage_types:
            if hierarchy["parameter_sweeps"] != "passed":
                return (False, "Parameter sweep validation must pass first")
        elif "ARRAY_SYSTEM" in plan_stage_types:
            if hierarchy["arrays_systems"] != "passed":
                return (False, "Array/system validation must pass first")
        elif "SINGLE_STRUCTURE" in plan_stage_types:
            if hierarchy["single_structure"] != "passed":
                return (False, "Single structure validation must pass first")
        return (True, None)
    
    # MATERIAL_VALIDATION and SINGLE_STRUCTURE only require material validation (already checked)
    return (True, None)


# ═══════════════════════════════════════════════════════════════════════
# User Interaction Logging
# ═══════════════════════════════════════════════════════════════════════

# Valid interaction types for log_user_interaction()
INTERACTION_TYPES = [
    "material_checkpoint",      # Mandatory Stage 0 approval
    "clarification",            # Ambiguous paper information
    "trade_off_decision",       # Accuracy vs runtime choices
    "parameter_confirmation",   # Key parameter values
    "stop_decision",            # Whether to stop reproduction
    "backtrack_approval",       # Approving suggested backtrack
    "context_overflow",         # Context overflow recovery decision
    "general_feedback",         # Other user input
]


def log_user_interaction(
    state: dict,
    interaction_type: str,
    question: str,
    response: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Log a user interaction for audit trail and reproducibility.
    
    MUST be called after every ask_user node completes, before transitioning
    to the next node. This is critical for:
    1. Audit trail - documenting what decisions were made and why
    2. Reproducibility - allowing reproduction of the decision path
    3. Report generation - showing user involvement in final report
    
    Args:
        state: ReproState dict (will be modified in place)
        interaction_type: Type of interaction (see INTERACTION_TYPES)
        question: The question that was posed to the user
        response: User's response dict (from user_responses field)
        context: Optional additional context (stage_id, agent, reason, etc.)
        
    Returns:
        The generated interaction ID (e.g., "U1", "U2")
        
    Raises:
        ValueError: If interaction_type is not valid
        
    Example:
        # After material checkpoint
        interaction_id = log_user_interaction(
            state,
            interaction_type="material_checkpoint",
            question=state["pending_user_questions"][0],
            response=state["user_responses"]["material_validation"],
            context={
                "stage_id": "stage0_material_validation",
                "agent": "SupervisorAgent"
            }
        )
        
    Example in ask_user node handler:
        def ask_user_node(state):
            # ... wait for user response ...
            
            # Log the interaction before transitioning
            log_user_interaction(
                state,
                interaction_type=state.get("pending_interaction_type", "general_feedback"),
                question=state["pending_user_questions"][0],
                response=state["user_responses"],
                context={"stage_id": state.get("current_stage_id")}
            )
            
            # Clear pending questions
            state["pending_user_questions"] = []
            state["awaiting_user_input"] = False
            
            return state
    """
    from datetime import datetime
    
    # Validate interaction type
    if interaction_type not in INTERACTION_TYPES:
        raise ValueError(
            f"Invalid interaction_type '{interaction_type}'. "
            f"Must be one of: {INTERACTION_TYPES}"
        )
    
    # Ensure user_interactions list exists
    if "user_interactions" not in state:
        state["user_interactions"] = []
    
    # Generate interaction ID
    interaction_id = f"U{len(state['user_interactions']) + 1}"
    
    # Build context with defaults
    full_context = {
        "stage_id": state.get("current_stage_id"),
        "agent": "SupervisorAgent",  # Default, can be overridden
        "reason": state.get("supervisor_feedback", "User input required"),
    }
    if context:
        full_context.update(context)
    
    # Create interaction record
    interaction_record = {
        "id": interaction_id,
        "timestamp": datetime.now().isoformat(),
        "interaction_type": interaction_type,
        "context": full_context,
        "question": question,
        "user_response": response,
        "impact": "",  # Filled after decision is applied
        "alternatives_considered": [],
    }
    
    # Append to state
    state["user_interactions"].append(interaction_record)
    
    # Also update progress if available (for persistence)
    if "progress" in state and isinstance(state["progress"], dict):
        if "user_interactions" not in state["progress"]:
            state["progress"]["user_interactions"] = []
        state["progress"]["user_interactions"].append(interaction_record)
    
    return interaction_id


def update_interaction_impact(
    state: dict,
    interaction_id: str,
    impact: str,
    alternatives_considered: Optional[List[str]] = None
) -> None:
    """
    Update a logged interaction with its impact after the decision is applied.
    
    Call this after applying the user's decision to document what effect
    it had on the workflow.
    
    Args:
        state: ReproState dict
        interaction_id: ID of the interaction to update (e.g., "U1")
        impact: Description of what changed as a result
        alternatives_considered: Optional list of alternatives that were presented
        
    Example:
        log_user_interaction(state, "material_checkpoint", question, response)
        # ... apply the decision ...
        update_interaction_impact(
            state, 
            "U1",
            impact="Changed material from silver to gold, invalidated Stage 0",
            alternatives_considered=["Keep silver", "Use aluminum"]
        )
    """
    for interaction in state.get("user_interactions", []):
        if interaction["id"] == interaction_id:
            interaction["impact"] = impact
            if alternatives_considered:
                interaction["alternatives_considered"] = alternatives_considered
            break
    
    # Also update in progress if available
    if "progress" in state and isinstance(state["progress"], dict):
        for interaction in state["progress"].get("user_interactions", []):
            if interaction["id"] == interaction_id:
                interaction["impact"] = impact
                if alternatives_considered:
                    interaction["alternatives_considered"] = alternatives_considered
                break


class MaterialValidationUserResponse(TypedDict):
    """
    User response to Stage 0 material validation checkpoint.
    
    This is the most critical checkpoint in the system. If materials are wrong,
    all downstream simulations will produce incorrect results.
    """
    verdict: str  # "approve" | "change_database" | "change_material" | "need_help"
    database_choice: NotRequired[str]  # e.g., "johnson_christy", "rakic" - if changing database
    material_choice: NotRequired[str]  # e.g., "gold" instead of "silver" - if changing material
    material_id: NotRequired[str]  # Which material to change (if changing database)
    notes: str  # User explanation/reasoning


class ValidatedMaterial(TypedDict):
    """
    A user-confirmed material for use in Stage 1+ simulations.
    
    This is populated ONLY by material_checkpoint_node after user approval.
    Immutable after creation per Data Ownership Matrix.
    """
    material_id: str  # e.g., "palik_gold"
    name: str  # e.g., "Gold"
    source: str  # e.g., "palik", "johnson_christy", "rakic"
    path: str  # Path to material data file
    validated_by_user: bool  # Always True when in validated_materials
    validation_timestamp: str  # ISO 8601 timestamp


def populate_validated_materials(
    state: dict,
    user_response: MaterialValidationUserResponse
) -> List[ValidatedMaterial]:
    """
    Populate validated_materials from planned_materials after user approval.
    
    This function is called by material_checkpoint_node when the user approves
    the materials. It converts planned_materials to validated_materials format.
    
    IMPORTANT: This is the ONLY place validated_materials should be populated.
    Stage 1+ CodeGeneratorAgent reads from validated_materials, not planned_materials.
    
    Args:
        state: ReproState dict containing planned_materials
        user_response: User's material checkpoint response (must have verdict="approve")
        
    Returns:
        List of ValidatedMaterial dicts ready to set on state["validated_materials"]
        
    Raises:
        ValueError: If user_response verdict is not "approve"
        
    Example:
        >>> if user_response["verdict"] == "approve":
        ...     validated = populate_validated_materials(state, user_response)
        ...     return {"validated_materials": validated}
    """
    if user_response.get("verdict") != "approve":
        raise ValueError(
            f"Cannot populate validated_materials: verdict is '{user_response.get('verdict')}', not 'approve'"
        )
    
    planned_materials = state.get("planned_materials", [])
    if not planned_materials:
        raise ValueError(
            "Cannot populate validated_materials: planned_materials is empty. "
            "PlannerAgent should have populated this during planning phase."
        )
    
    validated: List[ValidatedMaterial] = []
    timestamp = datetime.now().isoformat()
    
    for mat in planned_materials:
        validated.append(ValidatedMaterial(
            material_id=mat.get("material_id", "unknown"),
            name=mat.get("name", "Unknown"),
            source=mat.get("source", "unknown"),
            path=mat.get("path", ""),
            validated_by_user=True,
            validation_timestamp=timestamp,
        ))
    
    return validated


def get_material_path(state: dict, material_name: str) -> Optional[str]:
    """
    Get the file path for a validated material by name.
    
    This is a convenience function for CodeGeneratorAgent to look up material paths.
    
    Args:
        state: ReproState dict containing validated_materials
        material_name: Material name to find (case-insensitive)
        
    Returns:
        Path to material data file, or None if not found
        
    Example:
        >>> gold_path = get_material_path(state, "Gold")
        >>> if gold_path:
        ...     # Use gold_path in code generation
    """
    validated = state.get("validated_materials", [])
    name_lower = material_name.lower()
    
    for mat in validated:
        if mat.get("name", "").lower() == name_lower:
            return mat.get("path")
    
    return None


def check_materials_validated(state: dict) -> tuple:
    """
    Check if materials have been validated (required for Stage 1+).
    
    This should be called by CodeGeneratorAgent at the start of Stage 1+ code generation.
    If materials are not validated, code generation should fail with a clear error.
    
    Args:
        state: ReproState dict
        
    Returns:
        Tuple of (is_validated: bool, error_message: str or None)
        
    Example:
        >>> is_valid, error = check_materials_validated(state)
        >>> if not is_valid:
        ...     return {"run_error": error}
    """
    current_stage_type = state.get("current_stage_type", "")
    
    # Stage 0 doesn't need validated_materials - it creates them
    if current_stage_type == "MATERIAL_VALIDATION":
        return (True, None)
    
    validated = state.get("validated_materials", [])
    if not validated:
        return (False, 
            "validated_materials is empty but required for Stage 1+ code generation. "
            "This indicates material_checkpoint_node did not run or user did not approve materials. "
            "Check that Stage 0 completed and user confirmation was received."
        )
    
    return (True, None)


# ═══════════════════════════════════════════════════════════════════════
# Main State Definition
# ═══════════════════════════════════════════════════════════════════════
#
# NOTE: Types like ExtractedParameter, Discrepancy, StageProgress,
# FigureComparison, AgentCallMetric, StageMetric, MetricsLog, etc.
# are defined in the JSON schemas and should be generated from there.
#
# See the header of this file for generation instructions.
# ═══════════════════════════════════════════════════════════════════════

class ReproState(TypedDict, total=False):
    """
    Main state object for the paper reproduction LangGraph.
    
    This state flows through all nodes and maintains the complete
    context of the reproduction effort.
    """
    
    # ─── Paper Identification ───────────────────────────────────────────
    paper_id: str
    paper_domain: str  # plasmonics | photonic_crystal | metamaterial | thin_film | waveguide | strong_coupling | nonlinear | other
    paper_text: str  # Full extracted text from PDF
    paper_title: str
    paper_citation: Optional[Dict[str, Any]] # Added for report schema compatibility
    
    # ─── Runtime & Hardware Configuration ────────────────────────────────
    # These control execution behavior, timeouts, and resource limits.
    # Passed in via create_initial_state() or use defaults.
    runtime_config: "RuntimeConfig"  # Timeouts, debug mode, retry limits
    hardware_config: "HardwareConfig"  # CPU cores, RAM, GPU availability
    
    # ─── Shared Artifacts (mirrors of JSON files) ───────────────────────
    # These are kept in memory and periodically saved to disk
    plan: Dict[str, Any]  # Full plan structure
    assumptions: Dict[str, Any]  # Full assumptions structure
    progress: Dict[str, Any]  # Full progress structure
    
    # ─── Extracted Parameters ───────────────────────────────────────────
    # NOTE: Data Ownership Contract
    # 
    # The extracted_parameters field follows a specific sync pattern:
    #
    # 1. CANONICAL SOURCE: plan["extracted_parameters"]
    #    - Persisted to disk in plan_{paper_id}.json
    #    - PlannerAgent is the ONLY writer
    #    - Updated during PLAN node execution
    #
    # 2. IN-MEMORY VIEW: state.extracted_parameters
    #    - Typed in-memory copy for type safety
    #    - All agents (except Planner) READ from this field
    #    - Do NOT modify directly; changes won't persist
    #
    # 3. SYNC TIMING:
    #    - After PLAN node completes → sync from plan to state
    #    - At each checkpoint save → sync from plan to state
    #    - On checkpoint load → sync from loaded plan to state
    #
    # 4. SYNC IMPLEMENTATION:
    #    - Use sync_extracted_parameters(state) helper function
    #    - Performed automatically by workflow runner at sync points
    #
    # See sync_extracted_parameters() function below for implementation.
    # Structure matches plan_schema.json extracted_parameters items
    extracted_parameters: List[Dict[str, Any]]
    
    # ─── Materials (Two-Phase Handling) ─────────────────────────────────────
    # 
    # PHASE 1: planned_materials (populated by PlannerAgent)
    # - Contains materials identified from paper analysis
    # - Used by Stage 0 Code Generator (before validation)
    # - Source: PlannerAgent extracts from paper text/figures
    # Example: [{"material_id": "palik_gold", "name": "Gold", "path": "materials/palik_gold.csv"}]
    planned_materials: List[Dict[str, Any]]
    
    # PHASE 2: validated_materials (populated after Stage 0 + user confirmation)
    # - After material_checkpoint node, stores user-confirmed material mappings
    # - Used by Stage 1+ Code Generator (after validation)
    # - Code Generator for Stage 1+ MUST read from here, NOT hardcode paths
    # Example: [{"material_id": "palik_gold", "name": "Gold", "source": "palik", "path": "materials/palik_gold.csv"}]
    validated_materials: List[Dict[str, str]]
    
    # PENDING: Materials extracted during material_checkpoint but not yet approved
    # - Set by material_checkpoint_node
    # - Moved to validated_materials by supervisor_node on user approval
    # - Cleared after approval or rejection
    pending_validated_materials: List[Dict[str, str]]
    
    # ─── Validation Tracking ────────────────────────────────────────────
    # NOTE: validation_hierarchy is computed on demand via get_validation_hierarchy()
    # from progress["stages"]. It is not stored in state to avoid sync issues.
    geometry_interpretations: Dict[str, str]  # ambiguous_term → interpretation
    # Structure matches progress_schema.json#/definitions/discrepancy
    discrepancies_log: List[Dict[str, Any]]  # All discrepancies across all stages
    systematic_shifts: List[str]  # Known systematic shifts
    
    # ─── Current Control ────────────────────────────────────────────────
    current_stage_id: Optional[str]
    current_stage_type: Optional[str]  # MATERIAL_VALIDATION | SINGLE_STRUCTURE | etc.
    workflow_phase: str  # planning | design | pre_run_review | running | analysis | post_run_review | supervision | reporting
    workflow_complete: bool  # True when all stages complete and report generated
    # NOTE: review_context removed - now using separate review nodes (plan_review, design_review, code_review)
    
    # ─── Revision Tracking ──────────────────────────────────────────────
    design_revision_count: int
    code_revision_count: int  # Tracks code generation revisions (from code review feedback)
    execution_failure_count: int
    """Per-stage execution failure count.
    
    Incremented: Each time simulation crashes/fails at runtime (ExecutionValidatorAgent verdict = "fail")
    Reset to 0: 
        - When stage completes successfully (verdict = "pass" or "warning")
        - When user intervenes via ASK_USER and provides guidance
        - When moving to a new stage (counter is per-stage, not global)
    Limit: max_execution_failures (default: 2 per stage)
    
    Scoping: Counter is PER-STAGE. This prevents one problematic stage from
    poisoning the entire workflow. Each new stage starts fresh.
    """
    total_execution_failures: int
    """Global counter across all stages (for metrics/reporting only).
    
    Never resets during a reproduction run. Used for:
    - Final report statistics
    - Identifying papers that are particularly difficult to simulate
    - Comparing reproduction difficulty across papers
    """
    physics_failure_count: int
    """Per-stage physics sanity check failure count.
    
    Incremented: Each time PhysicsSanityAgent verdict = "fail"
    Reset to 0: 
        - When stage completes successfully
        - When user intervenes via ASK_USER
        - When moving to a new stage
    Limit: max_physics_failures (default: 2 per stage)
    """
    analysis_revision_count: int
    replan_count: int
    
    # ─── Backtracking Support ─────────────────────────────────────────────
    # When an agent detects a significant issue that invalidates earlier work,
    # it can suggest backtracking. SupervisorAgent decides whether to accept.
    backtrack_suggestion: Optional[Dict[str, Any]]  # {suggesting_agent, target_stage_id, reason, severity}
    invalidated_stages: List[str]  # Stage IDs marked as needing re-run
    backtrack_count: int  # Track number of backtracks (for limits)
    
    # ─── Verdicts ───────────────────────────────────────────────────────
    # Separate verdict fields for each review/validation type
    # Review verdicts (from reviewer agents):
    last_plan_review_verdict: Optional[str]  # approve | needs_revision (from PlanReviewerAgent)
    last_design_review_verdict: Optional[str]  # approve | needs_revision (from DesignReviewerAgent)
    last_code_review_verdict: Optional[str]  # approve | needs_revision (from CodeReviewerAgent)
    reviewer_issues: List[ReviewerIssue]
    # Validation verdicts (from validator agents - copied from agent output's "verdict" field):
    execution_verdict: Optional[str]  # pass | warning | fail (from ExecutionValidatorAgent)
    physics_verdict: Optional[str]  # pass | warning | fail | design_flaw (from PhysicsSanityAgent)
    comparison_verdict: Optional[str]  # approve | needs_revision (from ComparisonValidatorAgent)
    # Supervisor verdict:
    supervisor_verdict: Optional[str]  # ok_continue | replan_needed | change_priority | ask_user | backtrack_to_stage
    backtrack_decision: Optional[Dict[str, Any]]  # {accepted, target_stage_id, stages_to_invalidate, reason}
    
    # ─── Stage Working Data ─────────────────────────────────────────────
    code: Optional[str]  # Current Python+Meep code
    design_description: Optional[str]  # Natural language design
    performance_estimate: Optional[Dict[str, Any]]
    stage_outputs: Dict[str, Any]  # Filenames, stdout, etc.
    run_error: Optional[str]  # Capture simulation failures
    analysis_summary: Optional[Dict[str, Any]]  # Structured analyzer summary
    analysis_result_reports: List[Dict[str, Any]]  # Detailed per-result comparisons
    analysis_overall_classification: Optional[str]  # ResultsAnalyzer overall verdict
    quantitative_summary: List[Dict[str, Any]]  # Aggregated quantitative metrics for reporting
    
    # ─── Agent Feedback ─────────────────────────────────────────────────
    reviewer_feedback: Optional[str]  # Last reviewer feedback for revision
    supervisor_feedback: Optional[str]  # Last supervisor feedback
    planner_feedback: Optional[str]  # Feedback for replanning
    
    # ─── Performance Tracking ───────────────────────────────────────────
    runtime_budget_remaining_seconds: float
    total_runtime_seconds: float
    stage_start_time: Optional[str]  # ISO 8601
    
    # ─── User Interaction ───────────────────────────────────────────────
    pending_user_questions: List[str]
    user_responses: Dict[str, str]  # question → response (current session)
    awaiting_user_input: bool
    # Structure matches progress_schema.json#/definitions/user_interaction
    user_interactions: List[Dict[str, Any]]  # Full log of all user decisions/feedback
    
    # Resume context - helps agents understand what triggered ask_user
    ask_user_trigger: Optional[str]  # What caused ask_user (e.g., "code_review_limit", "material_checkpoint")
    last_node_before_ask_user: Optional[str]  # Which node triggered the ask_user
    
    # ─── Report Generation ──────────────────────────────────────────────
    # Structures match report_schema.json definitions
    figure_comparisons: List[Dict[str, Any]]  # Matches report_schema.json#/definitions/figure_comparison
    overall_assessment: List[Dict[str, Any]]  # Matches report_schema.json executive_summary.overall_assessment
    systematic_discrepancies_identified: List[Dict[str, Any]]  # Matches report_schema.json systematic_discrepancies
    report_conclusions: Optional[Dict[str, Any]]  # Matches report_schema.json conclusions
    final_report_markdown: Optional[str]  # Generated REPRODUCTION_REPORT.md
    
    # ─── Metrics Tracking ─────────────────────────────────────────────────
    # Structure matches metrics_schema.json
    metrics: Optional[Dict[str, Any]]  # Comprehensive metrics for monitoring and learning
    
    # ─── Paper Figures (for multimodal comparison) ────────────────────────
    paper_figures: List[Dict[str, str]]  # [{id, description, image_path}, ...]
    
    # ─── Prompt Adaptations (from PromptAdaptorAgent) ───────────────────────
    # Stores the modifications made by PromptAdaptorAgent for this paper
    prompt_adaptations: List[Dict[str, Any]]  # List of {target_agent, modification_type, content, ...}


# ═══════════════════════════════════════════════════════════════════════
# State Initialization
# ═══════════════════════════════════════════════════════════════════════

def create_initial_state(
    paper_id: str,
    paper_text: str,
    paper_domain: str = "other",
    runtime_budget_minutes: float = 120.0,
    hardware_config: Optional[HardwareConfig] = None,
    runtime_config: Optional[RuntimeConfig] = None
) -> ReproState:
    """
    Create initial state for a new paper reproduction.
    
    Args:
        paper_id: Unique identifier for the paper
        paper_text: Extracted text from the paper PDF
        paper_domain: Primary domain (plasmonics, photonic_crystal, etc.)
        runtime_budget_minutes: Total runtime budget in minutes
        hardware_config: Optional hardware configuration (CPU cores, RAM, etc.)
                        Defaults to DEFAULT_HARDWARE_CONFIG if not provided.
        runtime_config: Optional runtime configuration (timeouts, debug mode, etc.)
                       Defaults to DEFAULT_RUNTIME_CONFIG if not provided.
    
    Returns:
        Initialized ReproState ready for the planning phase
    
    Example:
        # Basic usage with defaults
        state = create_initial_state("paper_123", paper_text)
        
        # With custom hardware config
        state = create_initial_state(
            "paper_123", 
            paper_text,
            hardware_config=HardwareConfig(cpu_cores=16, ram_gb=64, gpu_available=True)
        )
        
        # Debug mode
        state = create_initial_state(
            "paper_123",
            paper_text, 
            runtime_config=DEBUG_RUNTIME_CONFIG
        )
    """
    # Use defaults if not provided
    hw_config = hardware_config if hardware_config is not None else DEFAULT_HARDWARE_CONFIG
    rt_config = runtime_config if runtime_config is not None else DEFAULT_RUNTIME_CONFIG
    
    return ReproState(
        # Paper identification
        paper_id=paper_id,
        paper_domain=paper_domain,
        paper_text=paper_text,
        paper_title="",
        
        # Runtime & hardware configuration
        runtime_config=rt_config,
        hardware_config=hw_config,
        
        # Shared artifacts (empty, to be filled by PlannerAgent)
        plan={},
        assumptions={},
        progress={},
        
        # Extracted parameters
        extracted_parameters=[],
        
        # Materials (two-phase handling)
        planned_materials=[],  # Populated by PlannerAgent, used by Stage 0
        validated_materials=[],  # Populated after Stage 0 + user confirmation, used by Stage 1+
        pending_validated_materials=[],  # Pending materials awaiting user approval
        
        # Validation tracking
        # NOTE: validation_hierarchy is computed on demand via get_validation_hierarchy()
        geometry_interpretations={},
        discrepancies_log=[],
        systematic_shifts=[],
        
        # Current control
        current_stage_id=None,
        current_stage_type=None,
        workflow_phase="planning",
        # NOTE: review_context removed - now using separate review nodes
        
        # Revision tracking
        design_revision_count=0,
        code_revision_count=0,
        execution_failure_count=0,
        total_execution_failures=0,
        physics_failure_count=0,
        analysis_revision_count=0,
        replan_count=0,
        
        # Backtracking support
        backtrack_suggestion=None,
        invalidated_stages=[],
        backtrack_count=0,
        
        # Verdicts (separate fields for each review/validation type)
        last_plan_review_verdict=None,
        last_design_review_verdict=None,
        last_code_review_verdict=None,
        reviewer_issues=[],
        execution_verdict=None,  # From ExecutionValidatorAgent
        physics_verdict=None,  # From PhysicsSanityAgent
        comparison_verdict=None,  # From ComparisonValidatorAgent
        supervisor_verdict=None,
        backtrack_decision=None,
        
        # Stage working data
        code=None,
        design_description=None,
        performance_estimate=None,
        stage_outputs={},
        run_error=None,
        analysis_summary=None,
        
        # Agent feedback
        reviewer_feedback=None,
        supervisor_feedback=None,
        planner_feedback=None,
        
        # Performance tracking
        runtime_budget_remaining_seconds=runtime_budget_minutes * 60,
        total_runtime_seconds=0.0,
        stage_start_time=None,
        
        # User interaction
        pending_user_questions=[],
        user_responses={},
        awaiting_user_input=False,
        user_interactions=[],  # Log of all user decisions/feedback
        ask_user_trigger=None,  # What caused ask_user
        last_node_before_ask_user=None,  # Which node triggered ask_user
        
        # Report generation
        figure_comparisons=[],
        overall_assessment=[],
        systematic_discrepancies_identified=[],
        report_conclusions=None,
        final_report_markdown=None,
        
        # Metrics tracking
        metrics={
            "paper_id": paper_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "total_duration_seconds": None,
            "final_status": "in_progress",
            "agent_calls": [],
            "stage_metrics": [],
            "revision_summary": {
                "total_design_revisions": 0,
                "total_code_revisions": 0,
                "total_analysis_revisions": 0,
                "replans": 0,
                "user_interventions": 0,
                "revision_reasons": []
            },
            "token_summary": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "by_agent": {},
                "by_model": {}
            },
            "prompt_adaptations": {
                "adaptations_count": 0,
                "agents_modified": [],
                "adaptation_types": {}
            },
            "reproduction_quality": {
                "figures_targeted": 0,
                "figures_reproduced": 0,
                "success_count": 0,
                "partial_count": 0,
                "failure_count": 0,
                "success_rate": 0.0
            }
        },
        
        # Paper figures (populated from PaperInput)
        paper_figures=[],
        
        # Prompt adaptations (populated by PromptAdaptorAgent)
        prompt_adaptations=[]
    )


# ═══════════════════════════════════════════════════════════════════════
# Hardware Configuration
# ═══════════════════════════════════════════════════════════════════════

class HardwareConfig(TypedDict):
    """
    Hardware configuration for simulation execution.
    Used by SimulationDesignerAgent for runtime estimates and
    by CodeGeneratorAgent for parallelization decisions.
    """
    cpu_cores: int  # Number of CPU cores available
    ram_gb: int  # RAM in gigabytes
    gpu_available: bool  # Whether GPU acceleration is available (for future use)


# Default hardware configuration (power laptop)
DEFAULT_HARDWARE_CONFIG = HardwareConfig(
    cpu_cores=8,
    ram_gb=32,
    gpu_available=False
)


# ═══════════════════════════════════════════════════════════════════════
# Runtime Configuration
# ═══════════════════════════════════════════════════════════════════════

class RuntimeConfig(TypedDict):
    """
    Runtime configuration for the reproduction workflow.
    These values control timeouts, limits, and error recovery behavior.
    """
    # Total time budget
    max_total_runtime_hours: float  # Default: 8.0
    max_stage_runtime_minutes: float  # Default: 60.0
    
    # User interaction
    user_response_timeout_hours: float  # Default: 24.0
    
    # Error recovery limits
    physics_retry_limit: int  # Default: 2
    llm_retry_limit: int  # Default: 5
    json_parse_retry_limit: int  # Default: 3
    consecutive_failure_limit: int  # Default: 2
    
    # LLM retry backoff (exponential: 1s, 2s, 4s, 8s, 16s)
    llm_retry_base_seconds: float  # Default: 1.0
    llm_retry_max_seconds: float  # Default: 16.0
    
    # Debug mode settings
    debug_mode: bool  # Default: False - enables diagnostic quick-check mode
    debug_resolution_factor: float  # Default: 0.5 - multiplier for resolution in debug mode
    debug_max_stages: int  # Default: 2 - max stages to run in debug mode (Stage 0 + Stage 1)
    
    # Backtracking limits (moved from constants for configurability)
    max_backtracks: int  # Default: 2 - limit total backtracks to prevent infinite loops
    
    # Execution failure limits (distinct from code revision limits)
    max_execution_failures: int  # Default: 2 - limit simulation runtime crashes before escalating
    max_physics_failures: int  # Default: 2 - limit physics sanity failures before escalating

    # Revision limits (configurable)
    max_replans: int
    max_design_revisions: int
    max_code_revisions: int
    max_analysis_revisions: int


# Default runtime configuration
DEFAULT_RUNTIME_CONFIG = RuntimeConfig(
    max_total_runtime_hours=8.0,
    max_stage_runtime_minutes=60.0,
    user_response_timeout_hours=24.0,
    physics_retry_limit=2,
    llm_retry_limit=5,
    json_parse_retry_limit=3,
    consecutive_failure_limit=2,
    llm_retry_base_seconds=1.0,
    llm_retry_max_seconds=16.0,
    debug_mode=False,
    debug_resolution_factor=0.5,
    debug_max_stages=2,
    max_backtracks=2,
    max_execution_failures=2,
    max_physics_failures=2,
    max_replans=2,
    max_design_revisions=3,
    max_code_revisions=3,
    max_analysis_revisions=2
)

# Debug mode runtime configuration preset
DEBUG_RUNTIME_CONFIG = RuntimeConfig(
    max_total_runtime_hours=0.5,  # 30 minutes max
    max_stage_runtime_minutes=10.0,  # 10 minutes per stage
    user_response_timeout_hours=1.0,  # 1 hour timeout in debug
    physics_retry_limit=1,  # Fewer retries in debug mode
    llm_retry_limit=3,
    json_parse_retry_limit=2,
    consecutive_failure_limit=1,
    llm_retry_base_seconds=1.0,
    llm_retry_max_seconds=8.0,
    debug_mode=True,
    debug_resolution_factor=0.5,
    debug_max_stages=2,
    max_backtracks=1,
    max_execution_failures=1,
    max_physics_failures=1,
    max_replans=1,
    max_design_revisions=1,
    max_code_revisions=1,
    max_analysis_revisions=1
)


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# Revision limits
MAX_DESIGN_REVISIONS = 3
MAX_CODE_REVISIONS = 3  # Code generation revisions per stage (from code review feedback)
MAX_EXECUTION_FAILURES = 2  # Simulation runtime failures (crashes, timeouts) before escalating
MAX_PHYSICS_FAILURES = 2  # Physics sanity check failures before escalating
MAX_ANALYSIS_REVISIONS = 2
MAX_REPLANS = 2
# Note: MAX_BACKTRACKS, MAX_EXECUTION_FAILURES, MAX_PHYSICS_FAILURES are now configurable via RuntimeConfig
# These constants are kept for backwards compatibility but prefer using RuntimeConfig
MAX_BACKTRACKS = 2  # Default limit; configurable via RuntimeConfig

# Default runtime budgets (in minutes)
DEFAULT_STAGE_BUDGETS = {
    "MATERIAL_VALIDATION": 5,
    "SINGLE_STRUCTURE": 15,
    "ARRAY_SYSTEM": 30,
    "PARAMETER_SWEEP": 60,
    "COMPLEX_PHYSICS": 120
}

# Discrepancy thresholds (canonical source of truth)
# Values are percentages: 2 means ±2%
DISCREPANCY_THRESHOLDS = {
    "resonance_wavelength": {"excellent": 2, "acceptable": 5, "investigate": 10},
    "linewidth": {"excellent": 10, "acceptable": 30, "investigate": 50},
    "q_factor": {"excellent": 10, "acceptable": 30, "investigate": 50},
    "transmission": {"excellent": 5, "acceptable": 15, "investigate": 30},
    "reflection": {"excellent": 5, "acceptable": 15, "investigate": 30},
    "field_enhancement": {"excellent": 20, "acceptable": 50, "investigate": 100},
    "effective_index": {"excellent": 1, "acceptable": 3, "investigate": 5}
}

# Context window limits (for paper length validation)
# Claude Opus 4.5 has 200K token context window
CONTEXT_WINDOW_LIMITS = {
    "model_max_tokens": 200000,  # Claude Opus 4.5 limit
    "system_prompt_reserve": 5000,  # Reserved for system prompts
    "state_context_reserve": 3000,  # Reserved for state context
    "response_reserve": 8000,  # Reserved for model response
    "safe_paper_tokens": 150000,  # Safe limit for paper text
    "chars_per_token_estimate": 4,  # Rough estimate for character counting
    "max_paper_chars": 600000,  # Hard limit for v1 (exits with error if exceeded)
}

# Paper length warning thresholds (in characters)
PAPER_LENGTH_THRESHOLDS = {
    "normal_max_chars": 50000,  # < 50K chars = normal
    "long_max_chars": 150000,  # 50K-150K chars = long (consider trimming)
    "very_long_max_chars": 600000,  # > 150K chars = very long (likely needs trimming)
}


# ═══════════════════════════════════════════════════════════════════════
# Context Window Loop Management
# ═══════════════════════════════════════════════════════════════════════
# 
# During revision loops (design→review→revise, code→review→revise), context
# can grow as feedback accumulates. These constants and functions help
# prevent context overflow.

LOOP_CONTEXT_LIMITS = {
    # Estimated tokens per revision loop component
    "design_description_avg_tokens": 2000,  # ~8K chars
    "code_avg_tokens": 3000,  # ~12K chars
    "reviewer_feedback_avg_tokens": 500,  # ~2K chars
    "analysis_summary_avg_tokens": 1500,  # ~6K chars
    
    # Context growth multiplier per loop iteration
    # Each iteration adds: previous feedback + new artifact
    "growth_factor_per_iteration": 1.3,  # 30% growth per loop
    
    # Safety thresholds
    "loop_context_warning_tokens": 15000,  # Warn if loop context exceeds this
    "loop_context_critical_tokens": 25000,  # Force summarization if exceeded
    
    # Maximum context to carry between loops
    "max_feedback_history_tokens": 3000,  # Truncate older feedback
    "max_iterations_with_full_context": 2,  # After this, summarize history
}


def estimate_loop_context_tokens(
    loop_type: str,
    iteration: int,
    current_artifact_chars: int = 0,
    feedback_history_chars: int = 0
) -> int:
    """
    Estimate total context tokens for a revision loop iteration.
    
    This helps detect when context is growing too large and summarization
    or truncation may be needed.
    
    Args:
        loop_type: "design", "code", or "analysis"
        iteration: Current iteration number (1-based)
        current_artifact_chars: Character count of current design/code/analysis
        feedback_history_chars: Character count of accumulated feedback
        
    Returns:
        Estimated total tokens for this loop iteration
        
    Example:
        >>> tokens = estimate_loop_context_tokens("code", iteration=2, 
        ...     current_artifact_chars=10000, feedback_history_chars=2000)
        >>> if tokens > LOOP_CONTEXT_LIMITS["loop_context_warning_tokens"]:
        ...     print("Consider summarizing feedback history")
    """
    limits = LOOP_CONTEXT_LIMITS
    chars_per_token = CONTEXT_WINDOW_LIMITS["chars_per_token_estimate"]
    
    # Base tokens for the artifact type
    base_tokens = {
        "design": limits["design_description_avg_tokens"],
        "code": limits["code_avg_tokens"],
        "analysis": limits["analysis_summary_avg_tokens"],
    }.get(loop_type, 2000)
    
    # Add actual artifact size if provided
    if current_artifact_chars > 0:
        artifact_tokens = current_artifact_chars // chars_per_token
    else:
        artifact_tokens = base_tokens
    
    # Feedback history tokens
    feedback_tokens = feedback_history_chars // chars_per_token
    
    # Growth factor for accumulated context
    growth = limits["growth_factor_per_iteration"] ** (iteration - 1)
    
    # Total estimate
    total = int((artifact_tokens + feedback_tokens) * growth)
    
    # Add base overhead (system prompt, state context)
    overhead = (
        CONTEXT_WINDOW_LIMITS["system_prompt_reserve"] +
        CONTEXT_WINDOW_LIMITS["state_context_reserve"]
    )
    
    return total + overhead


def check_loop_context_status(
    loop_type: str,
    iteration: int,
    current_artifact_chars: int = 0,
    feedback_history_chars: int = 0
) -> Dict[str, Any]:
    """
    Check if loop context is within safe limits and provide recommendations.
    
    Args:
        loop_type: "design", "code", or "analysis"
        iteration: Current iteration number (1-based)
        current_artifact_chars: Character count of current design/code/analysis
        feedback_history_chars: Character count of accumulated feedback
        
    Returns:
        Dict with:
        - status: "ok", "warning", or "critical"
        - estimated_tokens: Current estimated tokens
        - recommendation: Action to take (if any)
        - should_summarize: Whether to summarize feedback history
        
    Example:
        >>> status = check_loop_context_status("code", iteration=3,
        ...     current_artifact_chars=15000, feedback_history_chars=5000)
        >>> if status["should_summarize"]:
        ...     feedback = summarize_feedback_history(feedback_history)
    """
    limits = LOOP_CONTEXT_LIMITS
    
    estimated_tokens = estimate_loop_context_tokens(
        loop_type, iteration, current_artifact_chars, feedback_history_chars
    )
    
    result: Dict[str, Any] = {
        "estimated_tokens": estimated_tokens,
        "iteration": iteration,
        "loop_type": loop_type,
    }
    
    # Check against thresholds
    if estimated_tokens >= limits["loop_context_critical_tokens"]:
        result["status"] = "critical"
        result["recommendation"] = (
            f"Context critical ({estimated_tokens:,} tokens). "
            "Summarize feedback history before continuing. "
            "Consider keeping only most recent feedback."
        )
        result["should_summarize"] = True
        
    elif estimated_tokens >= limits["loop_context_warning_tokens"]:
        result["status"] = "warning"
        result["recommendation"] = (
            f"Context high ({estimated_tokens:,} tokens). "
            "Consider summarizing older feedback if more iterations needed."
        )
        result["should_summarize"] = iteration > limits["max_iterations_with_full_context"]
        
    else:
        result["status"] = "ok"
        result["recommendation"] = None
        result["should_summarize"] = False
    
    return result


def truncate_feedback_history(
    feedback_items: List[str],
    max_tokens: Optional[int] = None
) -> List[str]:
    """
    Truncate feedback history to stay within token limits.
    
    Keeps most recent feedback items, dropping oldest ones first.
    
    Args:
        feedback_items: List of feedback strings (oldest first)
        max_tokens: Maximum tokens to keep (default from LOOP_CONTEXT_LIMITS)
        
    Returns:
        Truncated list of feedback items (most recent preserved)
        
    Example:
        >>> history = ["Feedback 1...", "Feedback 2...", "Feedback 3..."]
        >>> truncated = truncate_feedback_history(history, max_tokens=1000)
    """
    if max_tokens is None:
        max_tokens = LOOP_CONTEXT_LIMITS["max_feedback_history_tokens"]
    
    chars_per_token = CONTEXT_WINDOW_LIMITS["chars_per_token_estimate"]
    max_chars = max_tokens * chars_per_token
    
    # Work backwards from most recent
    result = []
    total_chars = 0
    
    for feedback in reversed(feedback_items):
        feedback_chars = len(feedback)
        if total_chars + feedback_chars <= max_chars:
            result.insert(0, feedback)
            total_chars += feedback_chars
        else:
            # Can't fit any more
            break
    
    return result


def create_feedback_summary_prompt(feedback_items: List[str]) -> str:
    """
    Create a prompt for summarizing feedback history.
    
    When feedback history exceeds limits, this generates a prompt that can
    be sent to an LLM to create a condensed summary.
    
    Args:
        feedback_items: List of feedback strings to summarize
        
    Returns:
        Prompt string for LLM to generate summary
    """
    combined = "\n\n---\n\n".join(feedback_items)
    
    return f"""Summarize the following revision feedback history into a concise summary.
Keep only:
1. Unresolved issues that still need attention
2. Key constraints or requirements mentioned
3. Patterns in the feedback (recurring issues)

Drop:
- Issues that were already addressed
- Verbose explanations
- Redundant points

Feedback History:
{combined}

Provide a condensed summary (max 500 words):"""


# ═══════════════════════════════════════════════════════════════════════
# Context Overflow Recovery
# ═══════════════════════════════════════════════════════════════════════
#
# When context would exceed safe limits, these functions help determine
# recovery options. Key principle: functions return ACTIONS, they don't
# mutate state directly (LangGraph nodes should apply state changes).
#
# Recovery priority order:
# 1. Summarize feedback (safest, preserves paper content)
# 2. Truncate paper to Methods (more aggressive, may lose context)
# 3. Escalate to user (always available as last resort)
# ═══════════════════════════════════════════════════════════════════════

class ContextOverflowError(Exception):
    """
    Raised when context would exceed safe limits and recovery failed.
    
    This exception signals that:
    1. Estimated context exceeds CONTEXT_WINDOW_LIMITS["safe_paper_tokens"]
    2. Automatic recovery was attempted (if recovery_attempted=True)
    3. Manual intervention or user decision is required
    
    Attributes:
        estimated_tokens: How many tokens were estimated
        max_tokens: The safe limit that was exceeded
        node: Which node/agent would have overflowed
        recovery_attempted: Whether automatic recovery was tried
        available_actions: List of recovery actions still possible
    """
    
    def __init__(
        self,
        estimated_tokens: int,
        max_tokens: int,
        node: str,
        recovery_attempted: bool = False,
        available_actions: Optional[List[str]] = None
    ):
        self.estimated_tokens = estimated_tokens
        self.max_tokens = max_tokens
        self.node = node
        self.recovery_attempted = recovery_attempted
        self.available_actions = available_actions or []
        
        msg = (
            f"Context overflow in {node}: {estimated_tokens:,} tokens "
            f"exceeds safe limit of {max_tokens:,}."
        )
        if recovery_attempted:
            msg += " Automatic recovery was attempted but insufficient."
        if available_actions:
            msg += f" Available actions: {', '.join(available_actions)}"
            
        super().__init__(msg)


def estimate_context_for_node(
    state: "ReproState",
    node_name: str
) -> Dict[str, Any]:
    """
    Estimate total context tokens that will be sent to LLM for a specific node.
    
    Different nodes receive different subsets of state, so context size varies.
    This builds on the existing loop context estimation but adds node-specific
    knowledge about what each agent receives.
    
    Args:
        state: Current ReproState
        node_name: Node about to execute (e.g., "plan", "design", "generate_code")
        
    Returns:
        Dict with:
        - estimated_tokens: Total estimated tokens
        - breakdown: Dict showing token contribution by field
        - status: "ok", "warning", or "critical"
        - exceeds_limit: Boolean
        
    Example:
        >>> result = estimate_context_for_node(state, "design")
        >>> if result["exceeds_limit"]:
        ...     actions = get_context_recovery_actions(...)
    """
    chars_per_token = CONTEXT_WINDOW_LIMITS["chars_per_token_estimate"]
    safe_limit = CONTEXT_WINDOW_LIMITS["safe_paper_tokens"]
    
    breakdown: Dict[str, int] = {}
    
    # Base overhead (system prompt, response reserve)
    breakdown["system_overhead"] = (
        CONTEXT_WINDOW_LIMITS["system_prompt_reserve"] +
        CONTEXT_WINDOW_LIMITS["response_reserve"]
    )
    
    # Node-specific context
    if node_name in ["plan", "adapt_prompts"]:
        # These receive full paper text
        # Handle None explicitly - get() returns None if key exists with None value
        paper_text = state.get("paper_text") or ""
        breakdown["paper_text"] = len(paper_text) // chars_per_token
        
    elif node_name in ["design", "generate_code", "review_design", "review_code"]:
        # These receive design + feedback + relevant plan sections
        breakdown["design_description"] = len(state.get("design_description", "") or "") // chars_per_token
        breakdown["reviewer_feedback"] = len(state.get("reviewer_feedback", "") or "") // chars_per_token
        breakdown["simulation_code"] = len(state.get("simulation_code", "") or "") // chars_per_token
        
        # Plan context (stage-specific, not full plan)
        plan = state.get("plan", {})
        current_stage = state.get("current_stage_id", "")
        if plan and current_stage:
            # Estimate stage context as subset of plan
            breakdown["plan_context"] = 2000  # Typical stage context
        else:
            breakdown["plan_context"] = len(str(plan)) // chars_per_token
            
    elif node_name in ["analyze", "compare", "physics_sanity"]:
        # These receive outputs + figures
        breakdown["stage_outputs"] = 3000  # Typical outputs context
        breakdown["paper_figures"] = 2000  # Figure context
        summary = state.get("analysis_summary")
        if isinstance(summary, str):
            summary_len = len(summary)
        elif isinstance(summary, dict):
            import json
            summary_len = len(json.dumps(summary))
        else:
            summary_len = 0
        breakdown["analysis_summary"] = summary_len // chars_per_token
        
    elif node_name == "supervisor":
        # Supervisor receives summary of everything
        breakdown["progress_summary"] = len(str(state.get("progress", {}))) // chars_per_token
        breakdown["figure_comparisons"] = len(str(state.get("figure_comparisons", []))) // chars_per_token
        
    # Common context (assumptions, current stage info)
    breakdown["assumptions"] = len(str(state.get("assumptions", {}))) // chars_per_token
    breakdown["common_context"] = CONTEXT_WINDOW_LIMITS["state_context_reserve"]
    
    total_tokens = sum(breakdown.values())
    
    # Determine status
    warning_threshold = int(safe_limit * 0.8)  # 80% of limit
    if total_tokens >= safe_limit:
        status = "critical"
    elif total_tokens >= warning_threshold:
        status = "warning"
    else:
        status = "ok"
    
    return {
        "estimated_tokens": total_tokens,
        "breakdown": breakdown,
        "status": status,
        "exceeds_limit": total_tokens >= safe_limit,
        "safe_limit": safe_limit,
        "tokens_over": max(0, total_tokens - safe_limit),
    }


def get_context_recovery_actions(
    node_name: str,
    estimated_tokens: int,
    state: "ReproState"
) -> List[Dict[str, Any]]:
    """
    Determine which recovery actions are available for a context overflow.
    
    Returns a list of actions in priority order. Does NOT mutate state.
    The caller (typically a LangGraph node) decides which action to take
    and applies the corresponding state updates.
    
    This design follows LangGraph patterns where:
    - Utility functions analyze and recommend
    - Nodes apply state changes through return values
    
    Args:
        node_name: Node that would overflow
        estimated_tokens: Current estimated tokens
        state: Current state (read-only)
        
    Returns:
        List of recovery actions, each with:
        - action: Action identifier
        - description: Human-readable description
        - estimated_savings: Token savings estimate
        - requires_llm_call: Whether LLM call is needed
        - Additional action-specific fields
        
    Example:
        >>> actions = get_context_recovery_actions("design", 180000, state)
        >>> for action in actions:
        ...     if action["estimated_savings"] >= tokens_over:
        ...         apply_action(action)
        ...         break
    """
    safe_limit = CONTEXT_WINDOW_LIMITS["safe_paper_tokens"]
    tokens_over = estimated_tokens - safe_limit
    chars_per_token = CONTEXT_WINDOW_LIMITS["chars_per_token_estimate"]
    
    actions: List[Dict[str, Any]] = []
    
    # ─── Action 1: Summarize feedback (safest) ───────────────────────────
    # This preserves paper content and only condenses iteration history
    feedback = state.get("reviewer_feedback", "") or ""
    if len(feedback) > 2000:
        # Estimate savings: keep ~500 chars worth of summary
        current_tokens = len(feedback) // chars_per_token
        after_tokens = 500 // chars_per_token
        savings = current_tokens - after_tokens
        
        actions.append({
            "action": "summarize_feedback",
            "priority": 1,
            "description": f"Summarize reviewer feedback ({len(feedback):,} chars → ~500 chars)",
            "estimated_savings": savings,
            "requires_llm_call": True,
            "prompt": create_feedback_summary_prompt([feedback]),
            "state_field": "reviewer_feedback",
            "risk_level": "low",
        })
    
    # ─── Action 2: Truncate old feedback history ─────────────────────────
    # Less aggressive than summarization, just drops old iterations
    if len(feedback) > 4000:
        # Keep only last 2000 chars
        current_tokens = len(feedback) // chars_per_token
        after_tokens = 2000 // chars_per_token
        savings = current_tokens - after_tokens
        
        actions.append({
            "action": "truncate_feedback",
            "priority": 2,
            "description": f"Keep only recent feedback ({len(feedback):,} → 2000 chars)",
            "estimated_savings": savings,
            "requires_llm_call": False,
            "truncation_rule": "keep_last_2000_chars",
            "state_field": "reviewer_feedback",
            "risk_level": "low",
        })
    
    # ─── Action 3: Use methods section only ──────────────────────────────
    # More aggressive - may lose context from other sections
    paper_text = state.get("paper_text", "") or ""
    if len(paper_text) > 30000:
        current_tokens = len(paper_text) // chars_per_token
        # Methods section typically 15-20K chars
        estimated_methods_tokens = 5000
        savings = current_tokens - estimated_methods_tokens
        
        actions.append({
            "action": "truncate_paper_to_methods",
            "priority": 3,
            "description": f"Keep only Methods section ({len(paper_text):,} chars → ~20K chars)",
            "estimated_savings": savings,
            "requires_llm_call": False,
            "extraction_note": "Use extract_methods_section() from paper_loader.py",
            "state_field": "paper_text",
            "risk_level": "medium",
            "warning": "May lose important context from Results/Discussion sections",
        })
    
    # ─── Action 4: Clear non-critical working fields ─────────────────────
    # Clear fields that can be regenerated
    clearable_fields = []
    clearable_savings = 0
    
    # Only suggest clearing fields that aren't needed for current node
    if node_name not in ["analyze", "compare"]:
        analysis_summary_val = state.get("analysis_summary")
        if isinstance(analysis_summary_val, dict):
            import json
            analysis_summary = json.dumps(analysis_summary_val)
        else:
            analysis_summary = analysis_summary_val or ""
        if len(analysis_summary) > 1000:
            clearable_fields.append("analysis_summary")
            clearable_savings += len(analysis_summary) // chars_per_token
    
    if clearable_fields and clearable_savings > 500:
        actions.append({
            "action": "clear_working_fields",
            "priority": 4,
            "description": f"Clear regenerable fields: {', '.join(clearable_fields)}",
            "estimated_savings": clearable_savings,
            "requires_llm_call": False,
            "fields_to_clear": clearable_fields,
            "risk_level": "medium",
            "warning": "These fields will need to be regenerated if needed later",
        })
    
    # ─── Action 5: Escalate to user (always available) ───────────────────
    actions.append({
        "action": "escalate_to_user",
        "priority": 99,  # Always last
        "description": "Ask user how to proceed",
        "estimated_savings": 0,  # Depends on user choice
        "requires_llm_call": False,
        "user_question": _build_overflow_user_question(
            node_name, estimated_tokens, safe_limit, actions
        ),
        "risk_level": "none",
        "interrupt_type": "ask_user",
    })
    
    # Sort by priority
    actions.sort(key=lambda x: x["priority"])
    
    return actions


def _build_overflow_user_question(
    node_name: str,
    estimated_tokens: int,
    safe_limit: int,
    available_actions: List[Dict[str, Any]]
) -> str:
    """Build the user question for escalation action."""
    
    options = []
    for i, action in enumerate(available_actions, 1):
        if action["action"] == "escalate_to_user":
            continue
        risk = action.get("risk_level", "unknown")
        savings = action.get("estimated_savings", 0)
        options.append(f"{i}. {action['description']} (saves ~{savings:,} tokens, risk: {risk})")
    
    options.append(f"{len(options) + 1}. Skip this stage and continue")
    options.append(f"{len(options) + 2}. Stop reproduction")
    
    return f"""Context overflow detected in {node_name}.

**Current estimate:** {estimated_tokens:,} tokens
**Safe limit:** {safe_limit:,} tokens  
**Over by:** {estimated_tokens - safe_limit:,} tokens

Available recovery options:
{chr(10).join(options)}

Which option should we use? (Enter number or describe custom approach)"""


def check_context_before_node(
    state: "ReproState",
    node_name: str,
    auto_recover: bool = True
) -> Dict[str, Any]:
    """
    Check context limits before executing a node and optionally auto-recover.
    
    This is the main entry point for context management. Call this at the
    start of each LangGraph node that makes LLM calls.
    
    Args:
        state: Current ReproState
        node_name: Node about to execute
        auto_recover: If True, attempt automatic recovery for low-risk actions
        
    Returns:
        Dict with:
        - ok: Boolean, True if safe to proceed
        - estimation: Full estimation result
        - recovery_applied: None or the action that was applied
        - state_updates: Dict of state changes to apply (if recovery needed)
        - escalate: Boolean, True if user intervention needed
        
    Example (in a LangGraph node):
        >>> def design_node(state):
        ...     check = check_context_before_node(state, "design")
        ...     if check["escalate"]:
        ...         return {
        ...             "pending_user_questions": [check["user_question"]],
        ...             "awaiting_user_input": True,
        ...         }
        ...     if check["state_updates"]:
        ...         state = {**state, **check["state_updates"]}
        ...     # Continue with normal node logic...
    """
    estimation = estimate_context_for_node(state, node_name)
    
    # ═══════════════════════════════════════════════════════════════════════
    # BASIC TOKEN TRACKING: Track cumulative token usage
    # ═══════════════════════════════════════════════════════════════════════
    # Initialize metrics if not present (don't mutate state directly, use state_updates)
    metrics = state.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    
    # Track estimated tokens for this call
    estimated_tokens = estimation.get("estimated_tokens", 0)
    current_budget = metrics.get("context_budget_used", 0)
    new_budget = current_budget + estimated_tokens
    
    # Update metrics dict (will be merged into state via state_updates)
    updated_metrics = {
        **metrics,
        "context_budget_used": new_budget,
    }
    
    # Ensure required fields exist
    if "agent_calls" not in updated_metrics:
        updated_metrics["agent_calls"] = []
    if "stage_metrics" not in updated_metrics:
        updated_metrics["stage_metrics"] = []
    if "total_input_tokens" not in updated_metrics:
        updated_metrics["total_input_tokens"] = 0
    if "total_output_tokens" not in updated_metrics:
        updated_metrics["total_output_tokens"] = 0
    
    # Log warning if approaching limits
    if new_budget > CONTEXT_WINDOW_LIMITS["safe_paper_tokens"] * 0.8:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Context budget usage high: {new_budget:,} tokens "
            f"({new_budget / CONTEXT_WINDOW_LIMITS['safe_paper_tokens'] * 100:.1f}% of safe limit)"
        )
    
    result: Dict[str, Any] = {
        "ok": True,
        "estimation": estimation,
        "recovery_applied": None,
        "state_updates": {
            "metrics": updated_metrics,  # Include updated metrics
        },
        "escalate": False,
        "user_question": None,
    }
    
    if estimation["status"] == "ok":
        return result
    
    if estimation["status"] == "warning":
        # Log warning but proceed
        import logging
        logging.getLogger(__name__).warning(
            f"Context warning in {node_name}: {estimation['estimated_tokens']:,} tokens "
            f"({estimation['status']})"
        )
        return result
    
    # Critical - need recovery
    result["ok"] = False
    
    actions = get_context_recovery_actions(
        node_name, 
        estimation["estimated_tokens"],
        state
    )
    
    if not auto_recover:
        # Return actions for caller to decide
        result["available_actions"] = actions
        return result
    
    # Attempt auto-recovery with low-risk actions
    for action in actions:
        if action.get("risk_level") != "low":
            continue
        if action.get("requires_llm_call"):
            continue  # Can't auto-apply LLM-dependent actions
            
        # Apply this action
        if action["action"] == "truncate_feedback":
            feedback = state.get("reviewer_feedback", "") or ""
            result["state_updates"]["reviewer_feedback"] = feedback[-2000:]
            result["recovery_applied"] = action
            result["ok"] = True
            
            import logging
            logging.getLogger(__name__).info(
                f"Auto-recovered from context overflow: {action['description']}"
            )
            return result
    
    # No auto-recovery worked - escalate to user
    escalate_action = next(
        (a for a in actions if a["action"] == "escalate_to_user"),
        None
    )
    
    if escalate_action:
        result["escalate"] = True
        result["user_question"] = escalate_action["user_question"]
        result["available_actions"] = actions
    
    return result


def format_thresholds_table() -> str:
    """
    Generate a markdown table from DISCREPANCY_THRESHOLDS.
    
    This function is used by the prompt builder to inject the canonical
    thresholds into agent prompts at runtime, ensuring single source of truth.
    
    Returns:
        Markdown table string with thresholds
    """
    # Human-readable names for quantities
    quantity_names = {
        "resonance_wavelength": "Resonance wavelength",
        "linewidth": "Linewidth / FWHM",
        "q_factor": "Q-factor",
        "transmission": "Transmission",
        "reflection": "Reflection",
        "field_enhancement": "Field enhancement",
        "effective_index": "Mode effective index",
    }
    
    rows = [
        "| Quantity | Excellent | Acceptable | Investigate |",
        "|----------|-----------|------------|-------------|"
    ]
    
    for key, thresholds in DISCREPANCY_THRESHOLDS.items():
        name = quantity_names.get(key, key.replace("_", " ").title())
        excellent = thresholds["excellent"]
        acceptable = thresholds["acceptable"]
        investigate = thresholds["investigate"]
        
        # Format: ±X% for excellent/acceptable, >X% for investigate
        rows.append(
            f"| {name} | ±{excellent}% | ±{acceptable}% | >{investigate}% |"
        )
    
    return "\n".join(rows)


def get_threshold_for_quantity(quantity: str, level: str = "acceptable") -> float:
    """
    Get the threshold value for a specific quantity and level.
    
    Args:
        quantity: Key from DISCREPANCY_THRESHOLDS (e.g., "resonance_wavelength")
        level: "excellent", "acceptable", or "investigate"
        
    Returns:
        Threshold as a decimal (e.g., 0.05 for 5%)
        
    Raises:
        KeyError: If quantity or level not found
    """
    return DISCREPANCY_THRESHOLDS[quantity][level] / 100.0

# Stage types in validation hierarchy order
STAGE_TYPE_ORDER = [
    "MATERIAL_VALIDATION",
    "SINGLE_STRUCTURE", 
    "ARRAY_SYSTEM",
    "PARAMETER_SWEEP",
    "COMPLEX_PHYSICS"
]


# ═══════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════

# Checkpoint locations in the workflow
CHECKPOINT_LOCATIONS = [
    "after_plan",           # After PlannerAgent completes
    "after_stage_complete", # After each stage completes (SUPERVISOR node)
    "before_ask_user",      # Before pausing for user input
]

# Checkpoint naming convention: checkpoint_<paper_id>_<location>_<timestamp>.json


def checkpoint_name_for_stage(state: "ReproState", event: str) -> str:
    """
    Generate consistent checkpoint names for stage-related events.
    
    This helper ensures checkpoint names follow a consistent convention:
    - stage0_complete, stage1_complete, etc. for completed stages
    - stage0_user_confirm, etc. for user confirmations
    
    Args:
        state: Current ReproState
        event: Event type (e.g., "complete", "user_confirm")
        
    Returns:
        Checkpoint name like "stage0_complete" or "stage2_user_confirm"
        
    Examples:
        >>> state = {"current_stage_id": "stage0_material_validation"}
        >>> checkpoint_name_for_stage(state, "complete")
        'stage0_complete'
        
        >>> state = {"current_stage_id": "stage2_bare_disk_sweep"}
        >>> checkpoint_name_for_stage(state, "user_confirm")
        'stage2_user_confirm'
    """
    stage_id = state.get("current_stage_id") or "unknown"
    
    # Extract stage number from stage_id (e.g., "stage0_material_validation" → "0")
    match = re.match(r"stage(\d+)_", stage_id)
    if match:
        stage_num = match.group(1)
        return f"stage{stage_num}_{event}"
    
    # Fallback: use full stage_id if pattern doesn't match
    return f"{stage_id}_{event}"


def save_checkpoint(
    state: ReproState,
    checkpoint_name: str,
    output_dir: str = "outputs"
) -> str:
    """
    Save a checkpoint of the current state.
    
    Also creates a "latest" pointer to this checkpoint for easy access.
    On Unix systems, the "latest" pointer is a symlink (space-efficient).
    On Windows or if symlink creation fails, falls back to a copy.
    
    Args:
        state: Current ReproState to save
        checkpoint_name: Name for this checkpoint (e.g., "after_plan", "stage2_complete")
        output_dir: Base output directory
        
    Returns:
        Path to saved checkpoint file
        
    Note:
        Windows symlink creation may fail without admin privileges or Developer Mode.
        The function gracefully falls back to file copy in such cases.
    """
    import json
    import os
    import shutil
    from pathlib import Path
    
    paper_id = state.get("paper_id", "unknown")
    # Use microseconds to ensure uniqueness even when checkpoints are created rapidly
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    checkpoint_dir = Path(output_dir) / paper_id / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"checkpoint_{paper_id}_{checkpoint_name}_{timestamp}.json"
    filepath = checkpoint_dir / filename
    
    # Ensure file doesn't already exist (handle extremely rare microsecond collision)
    # If it does exist, add a counter suffix
    counter = 0
    original_filepath = filepath
    while filepath.exists():
        counter += 1
        base_name = original_filepath.stem
        filepath = checkpoint_dir / f"{base_name}_{counter}.json"
    
    # Convert state to JSON-serializable dict
    state_dict = dict(state)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, indent=2, default=str)
    
    # Create "latest" pointer for easy access
    # Try symlink first (more space-efficient), fall back to copy
    latest_path = checkpoint_dir / f"checkpoint_{checkpoint_name}_latest.json"
    
    # Remove existing latest pointer (may be symlink or file)
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    
    # Use actual filename (may have counter suffix if collision occurred)
    actual_filename = filepath.name
    
    symlink_created = False
    try:
        # Try to create a relative symlink (works better across systems)
        latest_path.symlink_to(actual_filename)
        symlink_created = True
    except (OSError, NotImplementedError):
        # Symlink creation failed (common on Windows without admin/Developer Mode)
        # Fall back to file copy
        pass
    
    if not symlink_created:
        # Fall back: copy the checkpoint file
        shutil.copy2(filepath, latest_path)
    
    return str(filepath)


def load_checkpoint(
    paper_id: str,
    checkpoint_name: str = "latest",
    output_dir: str = "outputs"
) -> Optional[ReproState]:
    """
    Load a checkpoint to resume a reproduction.
    
    Args:
        paper_id: Paper identifier
        checkpoint_name: Specific checkpoint name or "latest" for most recent
        output_dir: Base output directory
        
    Returns:
        Loaded ReproState or None if not found
    """
    import json
    from pathlib import Path
    
    checkpoint_dir = Path(output_dir) / paper_id / "checkpoints"
    
    if checkpoint_name == "latest":
        # Find most recent checkpoint
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return None
        # Sort by modification time, get most recent
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        filepath = latest
    else:
        # Look for specific checkpoint
        filepath = checkpoint_dir / f"checkpoint_{checkpoint_name}_latest.json"
        if not filepath.exists():
            # Try with timestamp pattern
            matches = list(checkpoint_dir.glob(f"checkpoint_{paper_id}_{checkpoint_name}_*.json"))
            if not matches:
                return None
            filepath = max(matches, key=lambda p: p.stat().st_mtime)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        state_dict = json.load(f)
    
    return state_dict


def list_checkpoints(paper_id: str, output_dir: str = "outputs") -> List[Dict[str, str]]:
    """
    List all available checkpoints for a paper.
    
    Args:
        paper_id: Paper identifier
        output_dir: Base output directory
        
    Returns:
        List of checkpoint info dicts with name, timestamp, path
    """
    from pathlib import Path
    
    checkpoint_dir = Path(output_dir) / paper_id / "checkpoints"
    
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    for cp_file in checkpoint_dir.glob("checkpoint_*.json"):
        if "_latest" in cp_file.name:
            continue  # Skip "latest" symlinks
        
        # Parse filename: checkpoint_<paper_id>_<name>_<timestamp>.json
        parts = cp_file.stem.split("_")
        if len(parts) >= 4:
            name = "_".join(parts[2:-2])  # Everything between paper_id and timestamp
            timestamp = "_".join(parts[-2:])  # Last two parts are timestamp
        else:
            name = cp_file.stem
            timestamp = "unknown"
        
        checkpoints.append({
            "name": name,
            "timestamp": timestamp,
            "path": str(cp_file),
            "size_kb": cp_file.stat().st_size / 1024
        })
    
    # Sort by timestamp (most recent first)
    checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return checkpoints


# ═══════════════════════════════════════════════════════════════════════
# State Validation Guards
# ═══════════════════════════════════════════════════════════════════════

# Required state fields for each node
# This helps catch malformed state early in the workflow
# NOTE: Node names must match graph.py node names (lowercase)
NODE_REQUIREMENTS: Dict[str, List[str]] = {
    "adapt_prompts": [
        "paper_id",
        "paper_text",
        "paper_domain",
    ],
    "plan": [
        "paper_id",
        "paper_text",
        "paper_domain",
        "paper_figures",
    ],
    "select_stage": [
        "paper_id",
        "plan",
        "progress",  # validation_hierarchy is computed from progress["stages"]
    ],
    "design": [
        "paper_id",
        "current_stage_id",
        "plan",
        "assumptions",
    ],
    # Separate review nodes with focused requirements
    "plan_review": [
        "paper_id",
        "plan",
        "assumptions",
    ],
    "design_review": [
        "paper_id",
        "current_stage_id",
        "design_description",
    ],
    "code_review": [
        "paper_id",
        "current_stage_id",
        "code",
        "design_description",
    ],
    "generate_code": [
        "paper_id",
        "current_stage_id",
        "design_description",
    ],
    "run_code": [
        "paper_id",
        "current_stage_id",
        "code",
    ],
    "execution_check": [
        "paper_id",
        "current_stage_id",
        "stage_outputs",
    ],
    "physics_check": [
        "paper_id",
        "current_stage_id",
        "stage_outputs",
        "code",
    ],
    "analyze": [
        "paper_id",
        "current_stage_id",
        "stage_outputs",
        "paper_figures",
        "plan",
    ],
    "comparison_check": [
        "paper_id",
        "current_stage_id",
        "figure_comparisons",
        "analysis_summary",
    ],
    "supervisor": [
        "paper_id",
        "plan",
        "progress",  # validation_hierarchy is computed from progress["stages"]
    ],
    "handle_backtrack": [
        "paper_id",
        "backtrack_decision",
        "progress",
    ],
    "ask_user": [
        "paper_id",
        "pending_user_questions",
    ],
    "generate_report": [
        "paper_id",
        "plan",
        "progress",
        "figure_comparisons",
        "assumptions",
    ],
}


def validate_plan_targets_precision(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate that targets with high precision requirements have digitized data.
    
    RULE: Targets with precision_requirement="excellent" (<2%) MUST have
    digitized_data_path set. Without digitized data, we cannot accurately
    measure sub-2% discrepancies using vision comparison alone.
    
    Args:
        plan: Plan dict containing targets
        
    Returns:
        List of validation issues, each with:
        - figure_id: The affected target
        - issue: Description of the problem
        - severity: "blocking" (must fix) or "warning" (should fix)
        - suggestion: How to resolve
        
    Example:
        >>> issues = validate_plan_targets_precision(state["plan"])
        >>> blocking = [i for i in issues if i["severity"] == "blocking"]
        >>> if blocking:
        ...     raise ValueError(f"Plan has blocking issues: {blocking}")
    """
    issues = []
    targets = plan.get("targets", [])
    
    for target in targets:
        figure_id = target.get("figure_id", "unknown")
        precision = target.get("precision_requirement", "good")
        digitized_path = target.get("digitized_data_path")
        
        # ─── BLOCKING: Excellent precision requires digitized data ────────
        if precision == "excellent" and not digitized_path:
            issues.append({
                "figure_id": figure_id,
                "issue": f"Target {figure_id} has precision_requirement='excellent' (<2%) but no digitized_data_path",
                "severity": "blocking",
                "suggestion": (
                    f"Either: (1) Provide digitized data via WebPlotDigitizer and set digitized_data_path, "
                    f"or (2) Downgrade precision_requirement to 'good' (5%) for vision-only comparison. "
                    f"Sub-2% accuracy cannot be reliably achieved with vision comparison alone."
                ),
            })
        
        # ─── WARNING: Good precision benefits from digitized data ─────────
        elif precision == "good" and not digitized_path:
            issues.append({
                "figure_id": figure_id,
                "issue": f"Target {figure_id} has precision_requirement='good' (5%) without digitized_data_path",
                "severity": "warning",
                "suggestion": (
                    f"Digitized data is recommended for 'good' precision to enable quantitative metrics. "
                    f"Vision comparison may achieve this but with less certainty."
                ),
            })
    
    return issues


def validate_plan_completeness(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Comprehensive validation of a reproduction plan.
    
    Checks:
    1. Precision/digitized data requirements
    2. Stage dependencies are valid
    3. Material validation stage exists
    4. Target coverage
    
    Args:
        plan: Plan dict to validate
        
    Returns:
        List of validation issues with severity and suggestions
        
    Example:
        >>> issues = validate_plan_completeness(state["plan"])
        >>> for issue in issues:
        ...     if issue["severity"] == "blocking":
        ...         print(f"BLOCKING: {issue['issue']}")
    """
    all_issues = []
    
    # ─── Check 1: Precision requirements ─────────────────────────────────
    precision_issues = validate_plan_targets_precision(plan)
    all_issues.extend(precision_issues)
    
    # ─── Check 2: Material validation stage exists ───────────────────────
    stages = plan.get("stages", [])
    stage_types = {s.get("stage_type") for s in stages}
    
    if "MATERIAL_VALIDATION" not in stage_types:
        all_issues.append({
            "figure_id": None,
            "issue": "Plan is missing MATERIAL_VALIDATION stage (Stage 0)",
            "severity": "blocking",
            "suggestion": "Every plan MUST start with material validation. Add a Stage 0.",
        })
    
    # ─── Check 3: Stage dependencies reference valid stages ──────────────
    stage_ids = {s.get("stage_id") for s in stages}
    
    for stage in stages:
        stage_id = stage.get("stage_id", "unknown")
        for dep in stage.get("dependencies", []):
            if dep not in stage_ids:
                all_issues.append({
                    "figure_id": None,
                    "issue": f"Stage '{stage_id}' depends on '{dep}' which doesn't exist in plan",
                    "severity": "blocking",
                    "suggestion": f"Either add stage '{dep}' or remove it from dependencies.",
                })
    
    # ─── Check 4: Targets are covered by stages ──────────────────────────
    target_ids = {t.get("figure_id") for t in plan.get("targets", [])}
    covered_targets = set()
    for stage in stages:
        covered_targets.update(stage.get("targets", []))
    
    uncovered = target_ids - covered_targets
    if uncovered:
        all_issues.append({
            "figure_id": None,
            "issue": f"Targets not covered by any stage: {uncovered}",
            "severity": "warning",
            "suggestion": "Add stages to cover these targets or remove them from targets list.",
        })
    
    return all_issues


def validate_state_for_node(state: ReproState, node_name: str) -> List[str]:
    """
    Check that state has all required fields for a specific node.
    
    This helps catch malformed state early, preventing cryptic errors
    from missing fields deep in agent logic.
    
    Args:
        state: Current ReproState to validate
        node_name: Name of the node about to execute
        
    Returns:
        List of missing field names (empty if all required fields present)
        
    Example:
        >>> missing = validate_state_for_node(state, "analyze")
        >>> if missing:
        ...     raise ValueError(f"State missing required fields for analyze: {missing}")
    """
    if node_name not in NODE_REQUIREMENTS:
        # Unknown node - no validation defined
        return []
    
    required = NODE_REQUIREMENTS[node_name]
    missing = []
    
    for field in required:
        if field not in state:
            missing.append(field)
        elif state.get(field) is None:
            # Field exists but is None - may be acceptable for some fields
            # Only flag as missing if it's a critical field
            if field in ["paper_id", "paper_text", "plan", "code", "stage_outputs"]:
                missing.append(f"{field} (is None)")
    
    # ─── Conditional Validation: validated_materials for Stage 1+ ─────────
    # After Stage 0 (MATERIAL_VALIDATION), subsequent stages MUST have
    # validated_materials populated for Code Generator to use correct paths.
    if node_name == "generate_code":
        current_stage_type = state.get("current_stage_type", "")
        if current_stage_type != "MATERIAL_VALIDATION":
            validated_materials = state.get("validated_materials", [])
            if not validated_materials:
                missing.append("validated_materials (required for Stage 1+ code generation)")
    
    # ─── Conditional Validation: Plan targets precision for plan_review ───
    # During plan review, validate that high-precision targets have digitized data.
    if node_name == "plan_review":
        plan = state.get("plan", {})
        if plan:
            precision_issues = validate_plan_targets_precision(plan)
            blocking = [i for i in precision_issues if i["severity"] == "blocking"]
            for issue in blocking:
                missing.append(f"PLAN_ISSUE: {issue['issue']}")
    
    return missing


def validate_state_transition(
    state: ReproState, 
    from_node: str, 
    to_node: str
) -> List[str]:
    """
    Validate that state is ready for transition between nodes.
    
    Checks both that required fields exist and that expected outputs
    from the previous node are present.
    
    Args:
        state: Current ReproState
        from_node: Node that just completed
        to_node: Node about to execute
        
    Returns:
        List of validation issues (empty if transition is valid)
    """
    issues = []
    
    # Check required fields for target node
    missing = validate_state_for_node(state, to_node)
    if missing:
        issues.extend([f"Missing field for {to_node}: {f}" for f in missing])
    
    # Check specific transition requirements
    # NOTE: Node names must match graph.py node names (lowercase)
    transition_checks = {
        ("plan", "plan_review"): [
            ("plan", "Plan not set after plan node"),
            ("assumptions", "Assumptions not set after plan node"),
        ],
        ("plan_review", "select_stage"): [
            ("last_plan_review_verdict", "Plan review verdict not set"),
        ],
        ("design", "design_review"): [
            ("design_description", "Design description not set after design node"),
        ],
        ("design_review", "generate_code"): [
            ("last_design_review_verdict", "Design review verdict not set"),
        ],
        ("generate_code", "code_review"): [
            ("code", "Code not set after generate_code node"),
        ],
        ("code_review", "run_code"): [
            ("last_code_review_verdict", "Code review verdict not set"),
        ],
        ("run_code", "execution_check"): [
            ("stage_outputs", "Stage outputs not set after run_code node"),
        ],
        ("analyze", "comparison_check"): [
            ("analysis_summary", "Analysis summary not set after analyze node"),
        ],
    }
    
    transition_key = (from_node, to_node)
    if transition_key in transition_checks:
        for field, message in transition_checks[transition_key]:
            if field not in state or state.get(field) is None:
                issues.append(message)
    
    return issues


# ═══════════════════════════════════════════════════════════════════════
# Extracted Parameters Sync
# ═══════════════════════════════════════════════════════════════════════

def sync_extracted_parameters(state: ReproState) -> ReproState:
    """
    Synchronize extracted_parameters from plan to state.
    
    The canonical source for extracted parameters is plan["extracted_parameters"].
    This function copies that data to state.extracted_parameters for type-safe
    access by agents.
    
    Call this function:
    - After PLAN node completes
    - After loading a checkpoint
    - Before each checkpoint save (ensures consistency)
    
    Args:
        state: ReproState to update (modified in place)
        
    Returns:
        The same state object (for chaining)
        
    Example:
        # After planning completes
        state = sync_extracted_parameters(state)
        
        # Or in workflow runner
        if current_node == "plan":
            state = sync_extracted_parameters(state)
    """
    plan = state.get("plan", {})
    plan_params = plan.get("extracted_parameters", [])
    
    # Copy parameters from plan to state
    # Structure matches plan_schema.json extracted_parameters items
    typed_params: List[Dict[str, Any]] = []
    
    for param in plan_params:
        if isinstance(param, dict):
            # Ensure required fields exist with defaults
            # Fields match plan_schema.json extracted_parameters items
            typed_param: Dict[str, Any] = {
                "name": param.get("name", "unnamed"),
                "value": param.get("value"),
                "unit": param.get("unit", ""),
                "source": param.get("source", "inferred"),
                "location": param.get("location", ""),
                "cross_checked": param.get("cross_checked", False),
                "discrepancy_notes": param.get("discrepancy_notes"),
            }
            typed_params.append(typed_param)
    
    state["extracted_parameters"] = typed_params
    
    return state


def get_extracted_parameter(
    state: ReproState,
    name: str,
    default: Any = None
) -> Any:
    """
    Get a specific extracted parameter by name.
    
    Convenience function for looking up parameter values.
    
    Args:
        state: Current ReproState
        name: Parameter name to find
        default: Value to return if not found
        
    Returns:
        Parameter value, or default if not found
        
    Example:
        disk_diameter = get_extracted_parameter(state, "disk_diameter", default=75)
    """
    params = state.get("extracted_parameters", [])
    
    for param in params:
        if param.get("name") == name:
            return param.get("value", default)
    
    return default


def list_extracted_parameters(
    state: ReproState,
    cross_checked_only: bool = False,
    source_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List all extracted parameters, optionally filtered by cross-check status or source.
    
    Args:
        state: Current ReproState
        cross_checked_only: If True, only return parameters that were cross-checked
        source_filter: Only return params from this source type
                      ("text", "figure_caption", "figure_axis", "supplementary", "inferred")
                          
    Returns:
        List of parameter dicts with name, value, unit, source, cross_checked
        
    Example:
        # Get all cross-checked parameters
        confirmed = list_extracted_parameters(state, cross_checked_only=True)
        
        # Get parameters extracted from figures
        from_figures = list_extracted_parameters(state, source_filter="figure_axis")
    """
    params = state.get("extracted_parameters", [])
    
    result = []
    for p in params:
        # Apply filters
        if cross_checked_only and not p.get("cross_checked", False):
            continue
        if source_filter and p.get("source") != source_filter:
            continue
        
        result.append({
            "name": p.get("name"),
            "value": p.get("value"),
            "unit": p.get("unit"),
            "source": p.get("source"),
            "cross_checked": p.get("cross_checked", False),
            "location": p.get("location"),
        })
    
    return result


# ═══════════════════════════════════════════════════════════════════════
# Progress Stages Initialization
# ═══════════════════════════════════════════════════════════════════════
#
# DESIGN PRINCIPLE: Separation of Plan vs Progress
#
# Plan stages (plan["stages"]) contain DESIGN SPECS - immutable after approval:
#   - expected_outputs, validation_criteria, runtime_budget_minutes
#   - complexity_class, fallback_strategy, max_revisions
#   - These define WHAT should happen
#
# Progress stages (progress["stages"]) contain EXECUTION STATE - mutable:
#   - status, outputs, discrepancies, issues, runtime_seconds
#   - These track WHAT actually happened
#
# Progress stages reference plan stages by stage_id. Use get_plan_stage()
# to look up design specs when needed - don't duplicate them in progress.
# ═══════════════════════════════════════════════════════════════════════

def initialize_progress_from_plan(state: ReproState) -> ReproState:
    """
    Initialize progress["stages"] from plan["stages"] after planning completes.
    
    This creates progress entries that REFERENCE plan stages by stage_id.
    Design specs (expected_outputs, validation_criteria, etc.) are NOT copied
    to progress - use get_plan_stage() to look them up when needed.
    
    This separation ensures:
    - Single source of truth for design specs (plan)
    - Single source of truth for execution state (progress)
    - No risk of drift between duplicated data
    
    MUST be called after plan_node completes and before select_stage_node runs.
    
    Args:
        state: ReproState to update (modified in place)
        
    Returns:
        The same state object (for chaining)
        
    Example:
        # In plan_node after LLM generates plan:
        state["plan"] = generated_plan
        state = initialize_progress_from_plan(state)
        state = sync_extracted_parameters(state)
    """
    plan = state.get("plan", {})
    plan_stages = plan.get("stages", [])
    paper_id = state.get("paper_id", "unknown")
    
    if not plan_stages:
        return state
    
    # Initialize progress structure if not present
    if "progress" not in state or not isinstance(state["progress"], dict):
        state["progress"] = {
            "paper_id": paper_id,
            "total_runtime_seconds": 0,
            "stages": [],
            "discrepancy_summary": {
                "total_discrepancies": 0,
                "blocking_discrepancies": 0,
                "systematic_shifts_identified": []
            },
            "user_interactions": []
        }
        
    # Initialize metrics if not present
    if "metrics" not in state:
        state["metrics"] = {
            "paper_id": paper_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "agent_calls": [],
            "stage_metrics": []
        }
    
    # Convert plan stages to progress stages
    # Only store EXECUTION STATE fields - design specs stay in plan
    progress_stages = []
    current_time = datetime.now(timezone.utc).isoformat()
    
    for plan_stage in plan_stages:
        stage_id = plan_stage.get("stage_id", "unknown")
        stage_type = plan_stage.get("stage_type", "SINGLE_STRUCTURE")
        
        # Create progress stage entry with ONLY execution-state fields
        # Reference plan by stage_id for design specs
        progress_stage = {
            # ─── Reference to Plan (immutable identifier) ────────────────
            "stage_id": stage_id,  # Use this to look up plan specs
            "stage_type": stage_type,  # Cached for validation hierarchy
            
            # ─── Status Tracking (execution state) ───────────────────────
            "status": "not_started",
            "invalidation_reason": None,
            "last_updated": current_time,
            "revision_count": 0,
            "runtime_seconds": 0,
            "summary": f"Planned: {plan_stage.get('description', '')}",
            
            # ─── Output Tracking (execution results) ─────────────────────
            "outputs": [],  # Actual output files produced
            "discrepancies": [],  # Discrepancies found during analysis
            "issues": [],  # Problems encountered
            "next_actions": [],  # Suggested follow-ups
        }
        
        progress_stages.append(progress_stage)
    
    state["progress"]["stages"] = progress_stages
    
    return state


def get_plan_stage(state: ReproState, stage_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a stage's DESIGN SPECS from the plan by stage_id.
    
    Use this to look up immutable design specs like expected_outputs,
    validation_criteria, runtime_budget_minutes, etc. These are NOT
    stored in progress to avoid duplication.
    
    Args:
        state: Current ReproState
        stage_id: Stage identifier to find
        
    Returns:
        Plan stage dict with design specs, or None if not found
        
    Example:
        >>> plan_stage = get_plan_stage(state, "stage1_single_disk")
        >>> expected_outputs = plan_stage.get("expected_outputs", [])
        >>> runtime_budget = plan_stage.get("runtime_budget_minutes", 30)
    """
    plan = state.get("plan") or {}
    stages = plan.get("stages") or []
    
    for stage in stages:
        if stage.get("stage_id") == stage_id:
            return stage
    
    return None


def get_stage_design_spec(
    state: ReproState,
    stage_id: str,
    spec_name: str,
    default: Any = None
) -> Any:
    """
    Get a specific design spec for a stage from the plan.
    
    Convenience function for looking up individual design specs.
    
    Args:
        state: Current ReproState
        stage_id: Stage identifier
        spec_name: Name of spec to retrieve (e.g., "expected_outputs", "runtime_budget_minutes")
        default: Value to return if stage or spec not found
        
    Returns:
        The spec value, or default if not found
        
    Example:
        >>> outputs = get_stage_design_spec(state, "stage1", "expected_outputs", [])
        >>> budget = get_stage_design_spec(state, "stage1", "runtime_budget_minutes", 30)
    """
    plan_stage = get_plan_stage(state, stage_id)
    if plan_stage is None:
        return default
    return plan_stage.get(spec_name, default)


def get_current_stage_specs(state: ReproState) -> Optional[Dict[str, Any]]:
    """
    Get design specs for the CURRENT stage being executed.
    
    Convenience function that combines current_stage_id lookup with
    plan stage retrieval.
    
    Args:
        state: Current ReproState (must have current_stage_id set)
        
    Returns:
        Plan stage dict with design specs, or None if no current stage
        
    Example:
        >>> specs = get_current_stage_specs(state)
        >>> if specs:
        ...     for output_spec in specs.get("expected_outputs", []):
        ...         generate_output(output_spec)
    """
    current_stage_id = state.get("current_stage_id")
    if not current_stage_id:
        return None
    return get_plan_stage(state, current_stage_id)


def get_progress_stage(state: ReproState, stage_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific stage from progress by stage_id.
    
    Args:
        state: Current ReproState
        stage_id: Stage identifier to find
        
    Returns:
        Stage dict or None if not found
    """
    progress = state.get("progress", {})
    stages = progress.get("stages", [])
    
    for stage in stages:
        if stage.get("stage_id") == stage_id:
            return stage
    
    return None


def update_progress_stage_status(
    state: ReproState,
    stage_id: str,
    status: str,
    summary: Optional[str] = None,
    invalidation_reason: Optional[str] = None
) -> ReproState:
    """
    Update the status of a specific stage in progress.
    
    Valid status values:
    - "not_started": Stage hasn't run yet
    - "in_progress": Currently running
    - "completed_success": Finished with good results
    - "completed_partial": Finished with acceptable gaps
    - "completed_failed": Finished but failed validation
    - "blocked": Cannot run (dependencies not met or budget exceeded)
    - "needs_rerun": Backtrack target, will re-execute
    - "invalidated": Results invalid due to backtrack of dependency
    
    Args:
        state: ReproState to update
        stage_id: Stage to update
        status: New status value
        summary: Optional updated summary
        invalidation_reason: Optional reason if status is needs_rerun/invalidated
        
    Returns:
        The same state object (for chaining)
    """
    progress = state.get("progress", {})
    stages = progress.get("stages", [])
    
    for stage in stages:
        if stage.get("stage_id") == stage_id:
            stage["status"] = status
            stage["last_updated"] = datetime.now(timezone.utc).isoformat()
            if summary:
                stage["summary"] = summary
            if invalidation_reason:
                stage["invalidation_reason"] = invalidation_reason
            break
    
    metrics = state.get("metrics", {})
    stage_metrics = metrics.get("stage_metrics") if isinstance(metrics, dict) else None
    if stage_metrics:
        for metric in stage_metrics:
            if metric.get("stage_id") == stage_id:
                metric["final_status"] = status
                metric["completed_at"] = metric.get("completed_at") or datetime.now(timezone.utc).isoformat()
                break
    
    return state


def archive_stage_outputs_to_progress(
    state: ReproState,
    stage_id: str,
    runtime_seconds: Optional[float] = None
) -> ReproState:
    """
    Archive current stage_outputs to the progress entry for a stage.
    
    This should be called when a stage completes (success or partial) to persist
    the output file references in progress before moving to the next stage.
    
    The stage_outputs field contains transient data that gets reset between stages.
    This function copies the relevant parts to progress.stages[].outputs for
    permanent storage.
    
    Args:
        state: ReproState containing stage_outputs to archive
        stage_id: Stage to update in progress
        runtime_seconds: Optional runtime to record (from stage_outputs or passed directly)
        
    Returns:
        The same state object (for chaining)
        
    Example:
        >>> # After supervisor approves a stage as completed_success:
        >>> archive_stage_outputs_to_progress(state, "stage1_single_disk")
        >>> update_progress_stage_status(state, "stage1_single_disk", "completed_success")
    """
    progress = state.get("progress", {})
    stages = progress.get("stages", [])
    stage_outputs = state.get("stage_outputs", {})
    
    # Get expected outputs from plan to resolve types accurately
    plan_stage = get_plan_stage(state, stage_id)
    expected_outputs = plan_stage.get("expected_outputs", []) if plan_stage else []
    
    # Build outputs list from stage_outputs files
    outputs = []
    files = stage_outputs.get("files", [])
    
    for file_path in files:
        # Create output entry matching progress_schema.json#/definitions/output
        filename = file_path if isinstance(file_path, str) else file_path.get("path", str(file_path))
        
        # Default to inference
        output_type = _infer_output_type(filename)
        
        # Try to improve type using plan expectations
        # We match loosely on filename pattern or exact match
        for expected in expected_outputs:
            pattern = expected.get("filename_pattern", "")
            # Simple expansion of placeholders
            expanded = pattern.replace("{paper_id}", state.get("paper_id", "")).replace("{stage_id}", stage_id)
            
            if filename == expanded or (pattern and pattern in filename):
                art_type = expected.get("artifact_type", "")
                if "plot" in art_type or "png" in art_type:
                    output_type = "plot"
                elif "log" in art_type:
                    output_type = "log"
                else:
                    output_type = "data"
                break
        
        output_entry = {
            "filename": filename,
            "type": output_type,
        }
        
        # Try to link to figure comparison data if available
        # Check state["figure_comparisons"] for matches with this file or stage
        comparisons = state.get("figure_comparisons", [])
        for comp in comparisons:
            if comp.get("stage_id") == stage_id:
                # If this output is a plot for a specific figure, try to match
                target_fig = comp.get("figure_id")
                if target_fig and target_fig in filename:
                    output_entry["target_figure"] = target_fig
                    output_entry["result_status"] = comp.get("classification", "unknown")
                    output_entry["comparison_notes"] = comp.get("summary", "")
                    
        outputs.append(output_entry)
    
    # Find and update the stage
    for stage in stages:
        if stage.get("stage_id") == stage_id:
            stage["outputs"] = outputs
            stage["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Record runtime if available
            actual_runtime = 0.0
            if runtime_seconds is not None:
                stage["runtime_seconds"] = runtime_seconds
                actual_runtime = runtime_seconds
            elif "runtime_seconds" in stage_outputs:
                stage["runtime_seconds"] = stage_outputs["runtime_seconds"]
                actual_runtime = stage_outputs["runtime_seconds"]
            
            # Update global total runtime
            state["total_runtime_seconds"] = state.get("total_runtime_seconds", 0.0) + actual_runtime
            
            # Also record to metrics
            if "metrics" in state and "stage_metrics" in state["metrics"]:
                # Check if metric exists for this stage to avoid dupes
                existing = next((m for m in state["metrics"]["stage_metrics"] if m["stage_id"] == stage_id), None)
                if not existing:
                    metric_entry = {
                        "stage_id": stage_id,
                        "stage_type": stage.get("stage_type", "unknown"),
                        "simulation_runtime_seconds": actual_runtime,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "final_status": "success" # Assumed, updated by logic
                    }
                    state["metrics"]["stage_metrics"].append(metric_entry)
                else:
                    # Update existing
                    existing["simulation_runtime_seconds"] = actual_runtime
                    existing["completed_at"] = datetime.now(timezone.utc).isoformat()
            break
            
    return state


def _infer_output_type(file_path: Any) -> str:
    """
    Infer output type from filename extension.
    
    Maps file extensions to output types defined in progress_schema.json.
    Returns: "data", "plot", or "log"
    """
    if isinstance(file_path, dict):
        file_path = file_path.get("path", "")
    
    path_str = str(file_path).lower()
    
    if path_str.endswith(".csv") or path_str.endswith(".h5") or path_str.endswith(".npz") or path_str.endswith(".json") or path_str.endswith(".npy") or path_str.endswith(".hdf5"):
        return "data"
    elif path_str.endswith(".png") or path_str.endswith(".jpg") or path_str.endswith(".jpeg") or path_str.endswith(".pdf"):
        return "plot"
    elif path_str.endswith(".log") or path_str.endswith(".txt"):
        return "log"
    
    return "data"

