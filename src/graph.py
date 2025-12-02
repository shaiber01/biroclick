"""
LangGraph State Machine for ReproLab

This module constructs the LangGraph state graph for the ReproLab
paper reproduction system. It defines the workflow nodes and edges.

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
"""

import os
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

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

# Import factory-generated routing functions
from src.routing import (
    route_after_plan_review,
    route_after_design_review,
    route_after_code_review,
    route_after_execution_check,
    route_after_physics_check,
    route_after_comparison_check,
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

def route_after_plan(state: ReproState) -> Literal["plan_review"]:
    """Route after planning to dedicated plan review.
    
    The plan goes through PLAN_REVIEW node before proceeding to stage selection.
    This ensures plan quality before execution begins.
    """
    # Save checkpoint after planning
    save_checkpoint(state, "after_plan")
    return "plan_review"


def route_after_select_stage(state: ReproState) -> Literal["design", "generate_report"]:
    """
    Route based on next stage selection.
    If a stage is selected, go to design.
    If no stage returned (None), generate report.
    """
    if state.get("current_stage_id"):
        return "design"
    return "generate_report"


# ═══════════════════════════════════════════════════════════════════════
# Complex Routing Function (too many special cases for factory)
# ═══════════════════════════════════════════════════════════════════════

def route_after_supervisor(state: ReproState) -> Literal["select_stage", "plan", "ask_user", "handle_backtrack", "generate_report", "material_checkpoint"]:
    """
    Route after supervisor decision.
    
    This function has complex logic that doesn't fit the factory pattern:
    - Multiple verdict types with different behaviors
    - Special handling for MATERIAL_VALIDATION stage type
    - should_stop flag checking
    - Checkpoint naming using stage context
    
    Includes mandatory material checkpoint after Stage 0 completes.
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
        if current_stage_type == "MATERIAL_VALIDATION":
            return "material_checkpoint"
        
        return "select_stage"
        
    elif verdict == "replan_needed":
        if state.get("replan_count", 0) < MAX_REPLANS:
            return "plan"
        else:
            save_checkpoint(state, "before_ask_user_replan_limit")
            return "ask_user"
            
    elif verdict == "ask_user":
        return "ask_user"
        
    elif verdict == "backtrack_to_stage":
        return "handle_backtrack"
        
    elif verdict == "all_complete":
        return "generate_report"
        
    return "ask_user"  # Fallback


# ═══════════════════════════════════════════════════════════════════════
# Graph Construction
# ═══════════════════════════════════════════════════════════════════════

def create_repro_graph():
    """
    Constructs the LangGraph state graph for ReproLab.
    
    The graph has three separate review nodes:
    - plan_review: Reviews reproduction plan before stage selection
    - design_review: Reviews simulation design before code generation
    - code_review: Reviews generated code before execution
    """
    workflow = StateGraph(ReproState)

    # Add Nodes
    workflow.add_node("adapt_prompts", adapt_prompts_node)
    workflow.add_node("plan", plan_node)
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
    workflow.add_edge("adapt_prompts", "plan")
    
    # Plan → Plan Review → Select Stage (or back to Plan)
    workflow.add_conditional_edges(
        "plan",
        route_after_plan,
        {"plan_review": "plan_review"}
    )
    
    workflow.add_conditional_edges(
        "plan_review",
        route_after_plan_review,
        {
            "select_stage": "select_stage",
            "plan": "plan",
            "ask_user": "ask_user"
        }
    )
    
    workflow.add_conditional_edges(
        "select_stage",
        route_after_select_stage,
        {
            "design": "design",
            "generate_report": "generate_report"
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
            "analyze": "analyze"
        }
    )
    
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "select_stage": "select_stage",
            "plan": "plan",
            "ask_user": "ask_user",
            "handle_backtrack": "handle_backtrack",
            "generate_report": "generate_report",
            "material_checkpoint": "material_checkpoint"
        }
    )
    
    workflow.add_edge("handle_backtrack", "select_stage")
    
    # Material checkpoint ALWAYS routes to ask_user (mandatory user confirmation)
    workflow.add_edge("material_checkpoint", "ask_user")
    
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

    # Initialize memory checkpointer
    checkpointer = MemorySaver()

    # Compile with interrupt_before for ask_user node
    # This pauses the graph BEFORE ask_user executes, allowing external code to:
    # 1. Inspect state["pending_user_questions"]
    # 2. Collect user input
    # 3. Resume with user_response in state
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["ask_user"]
    )
