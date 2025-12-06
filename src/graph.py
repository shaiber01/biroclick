"""
LangGraph State Machine for ReproLab

This module constructs the LangGraph state graph for the ReproLab
paper reproduction system. It defines the workflow nodes and edges.

═══════════════════════════════════════════════════════════════════════════════
ROUTING ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

All routers use a unified mechanism for user interaction routing:

1. The `with_trigger_check` wrapper checks `ask_user_trigger` FIRST
2. If trigger is set, router returns "ask_user" immediately
3. This ensures consistent handling across all routing paths

Custom routers in this file are decorated with @with_trigger_check to match
the behavior of factory-created routers.

═══════════════════════════════════════════════════════════════════════════════
ROUTING FUNCTIONS
═══════════════════════════════════════════════════════════════════════════════

Most verdict-based routing functions are defined in src/routing.py using a
factory pattern to eliminate code duplication. This module imports:
- route_after_plan_review
- route_after_design_review
- route_after_code_review
- route_after_execution_check
- route_after_physics_check
- route_after_comparison_check

The following routers are defined locally because they have unique logic:
- route_after_plan: Simple checkpoint, always routes to plan_review
- route_after_select_stage: Checks current_stage_id, not a verdict
- route_after_supervisor: Complex multi-verdict with special cases
- route_after_ask_user: Always routes to supervisor (defined inline)

All local routers are wrapped with @with_trigger_check for consistent behavior.
"""

import os
from typing import Literal, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from src.persistence import JsonCheckpointSaver

from schemas.state import (
    ReproState, 
    save_checkpoint, 
    checkpoint_name_for_stage,
    MAX_REPLANS,
)
from src.agents import (
    adapt_prompts_node,
    plan_node,
    plan_reviewer_node,
    select_stage_node,
    simulation_designer_node,
    design_reviewer_node,
    code_reviewer_node,
    code_generator_node,
    execution_validator_node,
    physics_sanity_node,
    results_analyzer_node,
    comparison_validator_node,
    supervisor_node,
    ask_user_node,
    generate_report_node as _generate_report_node,
    handle_backtrack_node,
    material_checkpoint_node,
)
from src.code_runner import run_code_node
from src.agents.user_options import validate_no_collisions

# Validate user options configuration at module load time
# This ensures we catch any keyword collisions early
validate_no_collisions()

# Import factory-generated routing functions and the trigger check wrapper
from src.routing import (
    route_after_plan_review,
    route_after_design_review,
    route_after_code_review,
    route_after_execution_check,
    route_after_physics_check,
    route_after_comparison_check,
    with_trigger_check,
)


# ═══════════════════════════════════════════════════════════════════════
# Report Node Wrapper
# ═══════════════════════════════════════════════════════════════════════

def generate_report_node_with_checkpoint(state: ReproState) -> Dict[str, Any]:
    """
    Wrapper around generate_report_node that saves final checkpoint.
    
    Ensures the final state (including report path) is always persisted
    for archival and debugging purposes.
    """
    # Call the actual report generation
    result = _generate_report_node(state)
    
    # Save final checkpoint with the complete state including report
    # Merge result into state for checkpointing
    final_state = {**state, **result}
    save_checkpoint(final_state, "final_report")
    
    return result


# ═══════════════════════════════════════════════════════════════════════
# Simple Routing Functions (not verdict-based)
# ═══════════════════════════════════════════════════════════════════════

@with_trigger_check
def route_after_plan(state: ReproState) -> Literal["plan_review", "ask_user"]:
    """Route after planning to dedicated plan review.
    
    The plan goes through PLAN_REVIEW node before proceeding to stage selection.
    This ensures plan quality before execution begins.
    
    Note: Wrapped with @with_trigger_check to handle ask_user_trigger.
    """
    # Save checkpoint after planning
    save_checkpoint(state, "after_plan")
    return "plan_review"


@with_trigger_check
def route_after_select_stage(state: ReproState) -> Literal["design", "generate_report", "ask_user"]:
    """
    Route based on next stage selection.
    If a stage is selected, go to design.
    If no stage returned (None), generate report.
    
    Note: Wrapped with @with_trigger_check to handle ask_user_trigger.
    """
    if state.get("current_stage_id"):
        return "design"
    return "generate_report"


# ═══════════════════════════════════════════════════════════════════════
# Complex Routing Function (too many special cases for factory)
# ═══════════════════════════════════════════════════════════════════════

@with_trigger_check
def route_after_supervisor(state: ReproState) -> Literal["select_stage", "planning", "ask_user", "handle_backtrack", "generate_report", "material_checkpoint", "analyze", "generate_code", "design", "code_review", "design_review", "plan_review"]:
    """
    Route after supervisor decision.
    
    This function has complex logic that doesn't fit the factory pattern:
    - Multiple verdict types with different behaviors
    - Special handling for MATERIAL_VALIDATION stage type
    - should_stop flag checking
    - Checkpoint naming using stage context
    
    Includes mandatory material checkpoint after Stage 0 completes.
    
    Note: Wrapped with @with_trigger_check to handle ask_user_trigger.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    verdict = state.get("supervisor_verdict")
    current_stage_type = state.get("current_stage_type", "")
    
    # ═══════════════════════════════════════════════════════════════════════
    # VALIDATE VERDICT EXISTS: Handle None or missing verdict
    # ═══════════════════════════════════════════════════════════════════════
    if verdict is None:
        logger.warning(
            "supervisor_verdict is None - supervisor may not have run or failed. "
            "Escalating to user for guidance."
        )
        save_checkpoint(state, "before_ask_user_supervisor_error")
        return "ask_user"
    
    if verdict == "ok_continue" or verdict == "change_priority":
        if state.get("should_stop"):
            return "generate_report"
        
        # Save checkpoint after stage completion using consistent naming
        checkpoint_name = checkpoint_name_for_stage(state, "complete")
        save_checkpoint(state, checkpoint_name)
        
        # MANDATORY: Material checkpoint after Stage 0 completes
        # IMPORTANT: Check if we just finished material validation checkpoint to prevent loop
        if current_stage_type == "MATERIAL_VALIDATION":
            # If materials have been validated (user approved), proceed to next stage.
            # Note: We check validated_materials rather than user_responses because
            # ask_user_node stores responses with question text as key, not trigger name.
            validated_materials = state.get("validated_materials", [])
            if validated_materials:
                return "select_stage"
            return "material_checkpoint"
        
        return "select_stage"
        
    elif verdict == "replan_needed":
        runtime_config = state.get("runtime_config") or {}
        max_replans = runtime_config.get("max_replans", MAX_REPLANS)
        if state.get("replan_count", 0) < max_replans:
            return "planning"
        else:
            save_checkpoint(state, "before_ask_user_replan_limit")
            return "ask_user"
    
    # User-initiated replan with guidance bypasses count check
    elif verdict == "replan_with_guidance":
        return "planning"
            
    elif verdict == "ask_user":
        return "ask_user"
        
    elif verdict == "backtrack_to_stage":
        return "handle_backtrack"
        
    elif verdict == "all_complete":
        return "generate_report"
    
    elif verdict == "retry_analyze":
        # User provided hint for analysis_limit - retry analysis with the hint
        return "analyze"
    
    elif verdict == "retry_generate_code":
        # User provided hint for code_review_limit - retry code generation with the hint
        return "generate_code"
    
    elif verdict == "retry_design":
        # User provided hint for design_review_limit - retry design with the hint
        return "design"
    
    elif verdict == "retry_code_review":
        # User provided guidance for reviewer_escalation from code_review
        return "code_review"
    
    elif verdict == "retry_design_review":
        # User provided guidance for reviewer_escalation from design_review
        return "design_review"
    
    elif verdict == "retry_plan_review":
        # User provided guidance for reviewer_escalation from plan_review
        return "plan_review"
        
    return "ask_user"  # Fallback


# ═══════════════════════════════════════════════════════════════════════
# Graph Construction
# ═══════════════════════════════════════════════════════════════════════

def create_repro_graph(checkpoint_dir: Optional[str] = None):
    """
    Constructs the LangGraph state graph for ReproLab.
    
    The graph has three separate review nodes:
    - plan_review: Reviews reproduction plan before stage selection
    - design_review: Reviews simulation design before code generation
    - code_review: Reviews generated code before execution
    
    Args:
        checkpoint_dir: Optional directory for persistent checkpoints.
            If provided, uses JsonCheckpointSaver for disk persistence
            (enables resume after process exit).
            If None, uses MemorySaver (in-memory only, no resume after exit).
    """
    workflow = StateGraph(ReproState)

    # Add Nodes
    workflow.add_node("adapt_prompts", adapt_prompts_node)
    workflow.add_node("planning", plan_node)
    workflow.add_node("plan_review", plan_reviewer_node)
    workflow.add_node("select_stage", select_stage_node)
    workflow.add_node("design", simulation_designer_node)
    workflow.add_node("design_review", design_reviewer_node)
    workflow.add_node("code_review", code_reviewer_node)
    workflow.add_node("generate_code", code_generator_node)
    workflow.add_node("run_code", run_code_node)
    workflow.add_node("execution_check", execution_validator_node)
    workflow.add_node("physics_check", physics_sanity_node)
    workflow.add_node("analyze", results_analyzer_node)
    workflow.add_node("comparison_check", comparison_validator_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("ask_user", ask_user_node)
    workflow.add_node("generate_report", generate_report_node_with_checkpoint)
    workflow.add_node("handle_backtrack", handle_backtrack_node)
    workflow.add_node("material_checkpoint", material_checkpoint_node)

    # Define Edges
    workflow.add_edge(START, "adapt_prompts")
    workflow.add_edge("adapt_prompts", "planning")
    
    # Plan → Plan Review → Select Stage (or back to Plan)
    workflow.add_conditional_edges(
        "planning",
        route_after_plan,
        {"plan_review": "plan_review", "ask_user": "ask_user"}
    )
    
    workflow.add_conditional_edges(
        "plan_review",
        route_after_plan_review,
        {
            "select_stage": "select_stage",
            "planning": "planning",
            "ask_user": "ask_user"
        }
    )
    
    workflow.add_conditional_edges(
        "select_stage",
        route_after_select_stage,
        {
            "design": "design",
            "generate_report": "generate_report",
            "ask_user": "ask_user"
        }
    )
    
    # Design → Design Review → Generate Code (or back to Design)
    workflow.add_edge("design", "design_review")
    
    workflow.add_conditional_edges(
        "design_review",
        route_after_design_review,
        {
            "generate_code": "generate_code",
            "design": "design",
            "ask_user": "ask_user"
        }
    )
    
    # Generate Code → Code Review → Run Code (or back to Generate Code)
    workflow.add_edge("generate_code", "code_review")
    
    workflow.add_conditional_edges(
        "code_review",
        route_after_code_review,
        {
            "run_code": "run_code",
            "generate_code": "generate_code",
            "ask_user": "ask_user"
        }
    )
    workflow.add_edge("run_code", "execution_check")
    
    workflow.add_conditional_edges(
        "execution_check",
        route_after_execution_check,
        {
            "physics_check": "physics_check",
            "generate_code": "generate_code",
            "ask_user": "ask_user"
        }
    )
    
    workflow.add_conditional_edges(
        "physics_check",
        route_after_physics_check,
        {
            "analyze": "analyze",
            "generate_code": "generate_code",
            "design": "design",  # design_flaw verdict routes here
            "ask_user": "ask_user"
        }
    )
    
    workflow.add_edge("analyze", "comparison_check")
    
    workflow.add_conditional_edges(
        "comparison_check",
        route_after_comparison_check,
        {
            "supervisor": "supervisor",
            "analyze": "analyze",
            "ask_user": "ask_user"  # Router can return ask_user on None/invalid verdict
        }
    )
    
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "select_stage": "select_stage",
            "planning": "planning",
            "ask_user": "ask_user",
            "handle_backtrack": "handle_backtrack",
            "generate_report": "generate_report",
            "material_checkpoint": "material_checkpoint",
            "analyze": "analyze",
            "generate_code": "generate_code",
            "design": "design",
            "code_review": "code_review",
            "design_review": "design_review",
            "plan_review": "plan_review",
        }
    )
    
    workflow.add_edge("handle_backtrack", "select_stage")
    
    # Material checkpoint routes based on whether materials are already validated
    def _route_after_material_checkpoint(state: ReproState) -> str:
        """
        Route after material checkpoint.
        
        If materials are already validated, skip ask_user and go to select_stage.
        Otherwise, route to ask_user for user confirmation.
        
        Note: Wrapped with with_trigger_check to handle ask_user_trigger.
        """
        validated_materials = state.get("validated_materials", [])
        if validated_materials:
            return "select_stage"
        return "ask_user"
    
    # Wrap with trigger check for consistent handling
    route_after_material_checkpoint = with_trigger_check(_route_after_material_checkpoint)
    
    workflow.add_conditional_edges(
        "material_checkpoint",
        route_after_material_checkpoint,
        {
            "ask_user": "ask_user",
            "select_stage": "select_stage"
        }
    )
    
    # Ask user resumes to Supervisor, who evaluates user feedback and decides next steps.
    # 
    # IMPORTANT: Before routing to ask_user, nodes should set these state fields:
    #   - ask_user_trigger: What caused the ask (e.g., "code_review_limit", "material_checkpoint")
    #   - last_node_before_ask_user: The node that triggered ask_user (e.g., "code_review")
    #
    # The Supervisor will use these fields (via resume_context) to understand:
    #   1. What the user was responding to
    #   2. Where to route next based on user's answer
    
    def route_after_ask_user(state: ReproState) -> str:
        """
        Route after user provides input.
        
        Always routes to Supervisor, who has full context (including resume_context)
        to make the appropriate next-step decision based on:
        - What triggered the ask_user (state["ask_user_trigger"])
        - User's response (state["user_responses"])
        - Overall progress and validation state
        """
        return "supervisor"

    workflow.add_conditional_edges(
        "ask_user",
        route_after_ask_user,
        {"supervisor": "supervisor"}
    )

    workflow.add_edge("generate_report", END)

    # Initialize checkpointer based on checkpoint_dir parameter
    # - JsonCheckpointSaver: Persists to disk, enables resume after process exit
    # - MemorySaver: In-memory only, loses state on process exit (for testing/dev)
    if checkpoint_dir:
        checkpointer = JsonCheckpointSaver(checkpoint_dir)
    else:
        checkpointer = MemorySaver()

    # Compile with checkpointer for interrupt support
    # The ask_user node uses interrupt() internally to pause for user input,
    # so we don't need interrupt_before. The node runs once and pauses mid-execution
    # when it calls interrupt(). Resume with Command(resume=user_response).
    return workflow.compile(checkpointer=checkpointer)
