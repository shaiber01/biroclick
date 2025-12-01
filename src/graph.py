
import os
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from schemas.state import ReproState, save_checkpoint
from src.agents import (
    adapt_prompts_node,
    plan_node,
    select_stage_node,
    simulation_designer_node,
    code_reviewer_node,
    code_generator_node,
    execution_validator_node,
    physics_sanity_node,
    results_analyzer_node,
    comparison_validator_node,
    supervisor_node,
    ask_user_node,
    generate_report_node,
    handle_backtrack_node
)
# Assuming run_code_node is in code_runner.py as per docs/workflow.md
from src.code_runner import run_code_node

def route_after_plan(state: ReproState) -> Literal["select_stage"]:
    """Route after planning is always to select stage."""
    # Save checkpoint after planning
    save_checkpoint(state, "after_plan")
    return "select_stage"

def route_after_select_stage(state: ReproState) -> Literal["design", "generate_report"]:
    """
    Route based on next stage selection.
    If a stage is selected, go to design.
    If no stage returned (None), generation report.
    """
    # The select_stage_node should set current_stage_id in state
    # If it returns None or doesn't set it, we assume done.
    # Note: state updates in langgraph are merged, so we check state keys.
    # But select_stage_node logic in docs returns the stage_id.
    # In this implementation, select_stage_node updates state["current_stage_id"].
    
    if state.get("current_stage_id"):
        return "design"
    return "generate_report"

def route_after_code_review(state: ReproState) -> Literal["generate_code", "run_code", "design", "ask_user"]:
    """
    Route based on reviewer verdict.
    Context: CodeReviewer reviews DESIGN first, then CODE.
    We need to know which phase we are in.
    """
    verdict = state.get("last_reviewer_verdict")
    revision_count = 0
    
    # Determine if we are in design or code phase based on state
    # This logic relies on the node that just ran.
    # Ideally state has a flag, or we infer.
    # Let's assume state["workflow_phase"] is updated by nodes.
    phase = state.get("workflow_phase", "")
    
    if phase == "design_review":
        revision_count = state.get("design_revision_count", 0)
        if verdict == "approve":
            return "generate_code"
        elif verdict == "needs_revision":
            if revision_count < 3: # MAX_DESIGN_REVISIONS
                return "design"
            else:
                return "ask_user"
                
    elif phase == "code_review":
        revision_count = state.get("code_revision_count", 0)
        if verdict == "approve":
            return "run_code"
        elif verdict == "needs_revision":
            if revision_count < 3: # MAX_CODE_REVISIONS
                return "generate_code"
            else:
                return "ask_user"
    
    # Fallback/Safety
    return "ask_user"

def route_after_execution_check(state: ReproState) -> Literal["physics_check", "generate_code", "ask_user"]:
    verdict = state.get("execution_verdict")
    if verdict in ["pass", "warning"]:
        return "physics_check"
    elif verdict == "fail":
        # If recoverable error (not explicitly defined in simple state, assume recoverable for now)
        # For now, fail goes to generate_code unless revisions exceeded (handled in node or here)
        if state.get("code_revision_count", 0) < 3:
             return "generate_code"
        else:
             return "ask_user"
    return "ask_user"

def route_after_physics_check(state: ReproState) -> Literal["analyze", "generate_code", "ask_user"]:
    verdict = state.get("physics_verdict")
    if verdict in ["pass", "warning"]:
        return "analyze"
    elif verdict == "fail":
        if state.get("code_revision_count", 0) < 3:
            return "generate_code"
        else:
            return "ask_user"
    return "ask_user"

def route_after_comparison_check(state: ReproState) -> Literal["supervisor", "analyze"]:
    verdict = state.get("comparison_verdict")
    if verdict == "approve":
        return "supervisor"
    elif verdict == "needs_revision":
        if state.get("analysis_revision_count", 0) < 2: # MAX_ANALYSIS_REVISIONS
            return "analyze"
        else:
            return "supervisor" # Proceed with flag
    return "supervisor"

def route_after_supervisor(state: ReproState) -> Literal["select_stage", "plan", "ask_user", "handle_backtrack", "generate_report"]:
    verdict = state.get("supervisor_verdict")
    
    if verdict == "ok_continue" or verdict == "change_priority":
        if state.get("should_stop"):
            return "generate_report"
        # Save checkpoint after stage completion
        save_checkpoint(state, f"stage_{state.get('current_stage_id')}_complete")
        return "select_stage"
        
    elif verdict == "replan_needed":
        if state.get("replan_count", 0) < 2: # MAX_REPLANS
            return "plan"
        else:
            return "ask_user"
            
    elif verdict == "ask_user":
        return "ask_user"
        
    elif verdict == "backtrack_to_stage":
        return "handle_backtrack"
        
    elif verdict == "all_complete":
        return "generate_report"
        
    return "ask_user" # Fallback

def create_repro_graph():
    """
    Constructs the LangGraph state graph for ReproLab.
    """
    workflow = StateGraph(ReproState)

    # Add Nodes
    workflow.add_node("adapt_prompts", adapt_prompts_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("select_stage", select_stage_node)
    workflow.add_node("design", simulation_designer_node)
    workflow.add_node("code_review", code_reviewer_node)
    workflow.add_node("generate_code", code_generator_node)
    workflow.add_node("run_code", run_code_node)
    workflow.add_node("execution_check", execution_validator_node)
    workflow.add_node("physics_check", physics_sanity_node)
    workflow.add_node("analyze", results_analyzer_node)
    workflow.add_node("comparison_check", comparison_validator_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("ask_user", ask_user_node)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("handle_backtrack", handle_backtrack_node)

    # Define Edges
    workflow.add_edge(START, "adapt_prompts")
    workflow.add_edge("adapt_prompts", "plan")
    
    workflow.add_conditional_edges(
        "plan",
        route_after_plan,
        {"select_stage": "select_stage"}
    )
    
    workflow.add_conditional_edges(
        "select_stage",
        route_after_select_stage,
        {
            "design": "design",
            "generate_report": "generate_report"
        }
    )
    
    workflow.add_edge("design", "code_review")
    
    workflow.add_conditional_edges(
        "code_review",
        route_after_code_review,
        {
            "generate_code": "generate_code",
            "run_code": "run_code",
            "design": "design",
            "ask_user": "ask_user"
        }
    )
    
    workflow.add_edge("generate_code", "code_review")
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
            "generate_report": "generate_report"
        }
    )
    
    workflow.add_edge("handle_backtrack", "select_stage")
    
    # Ask user always resumes to... depends on context.
    # The AskUser node should likely look at state to know where to resume.
    # Or we can have conditional edge from ask_user.
    # For simplicity in v1, ask_user often resumes based on what triggered it.
    # We can implement a router or just have it go to a "resume_router" node.
    # Actually, the graph topology usually requires fixed edges.
    # A common pattern is to have ask_user return to a specific node or use a "router" node.
    # Let's assume ask_user sets a "resume_node" in state or we use a router.
    
    def route_after_ask_user(state: ReproState) -> str:
        # Logic to determine where to go after user input
        # This depends on what triggered ask_user
        # This is complex; simplifying assumption: 
        # If triggered by stage 0 checkpoint -> select_stage (or design if re-doing)
        # If triggered by supervisor -> supervisor (to re-evaluate) or select_stage
        # If triggered by reviewer -> design or generate_code
        
        # For now, let's route back to Supervisor to re-evaluate the state with new user input
        # The supervisor can then decide next steps.
        return "supervisor"

    workflow.add_conditional_edges(
        "ask_user",
        route_after_ask_user,
        {"supervisor": "supervisor"}
    )

    workflow.add_edge("generate_report", END)

    # Initialize memory checkpointer
    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)

