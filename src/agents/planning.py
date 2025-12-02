"""
Planning agent nodes: plan_node, plan_reviewer_node, adapt_prompts_node.

These nodes handle the initial paper analysis and reproduction plan creation.

State Keys
----------
adapt_prompts_node:
    READS: paper_text, paper_domain
    WRITES: workflow_phase, prompt_adaptations, paper_domain

plan_node:
    READS: paper_text, paper_id, paper_figures, paper_domain, replan_count,
           runtime_config, prompt_adaptations
    WRITES: workflow_phase, plan, planned_materials, assumptions, paper_domain,
            progress, extracted_parameters, last_plan_review_verdict, planner_feedback,
            replan_count, ask_user_trigger, pending_user_questions, awaiting_user_input

plan_reviewer_node:
    READS: plan, paper_text, paper_figures, paper_domain, assumptions,
           last_plan_review_verdict, planner_feedback, digitized_data, replan_count,
           runtime_config
    WRITES: workflow_phase, last_plan_review_verdict, planner_feedback, replan_count,
            ask_user_trigger, pending_user_questions, awaiting_user_input
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

from schemas.state import (
    ReproState,
    initialize_progress_from_plan,
    sync_extracted_parameters,
    MAX_REPLANS,
)
from src.prompts import build_agent_prompt
from src.llm_client import (
    call_agent_with_metrics,
    build_user_content_for_planner,
)

from .helpers.context import check_context_or_escalate, validate_state_or_warn
from .helpers.metrics import log_agent_call
from .base import (
    with_context_check,
    create_llm_error_auto_approve,
    create_llm_error_escalation,
)


@with_context_check("adapt_prompts")
def adapt_prompts_node(state: ReproState) -> Dict[str, Any]:
    """PromptAdaptorAgent: Customize prompts for paper-specific needs.
    
    Note: Context check is handled by @with_context_check decorator.
    """
    # Build system prompt for prompt adaptor
    system_prompt = build_agent_prompt("prompt_adaptor", state)
    
    # Build user content with paper summary
    paper_text = state.get("paper_text", "")[:5000]  # First 5k chars for context
    paper_domain = state.get("paper_domain", "")
    
    user_content = f"# PAPER SUMMARY FOR PROMPT ADAPTATION\n\n"
    user_content += f"Domain: {paper_domain}\n\n"
    user_content += f"Paper excerpt:\n{paper_text[:3000]}...\n\n"
    user_content += "Analyze this paper and suggest prompt adaptations for better simulation reproduction."
    
    # Return state updates instead of mutating state
    result: Dict[str, Any] = {
        "workflow_phase": "adapting_prompts",
        "prompt_adaptations": [],  # Initialize empty adaptations list
    }
    
    # Call LLM for prompt adaptations
    try:
        agent_output = call_agent_with_metrics(
            agent_name="prompt_adaptor",
            system_prompt=system_prompt,
            user_content=user_content,
            state=state,
        )
        
        # Extract adaptations from agent output
        adaptations = agent_output.get("adaptations", [])
        result["prompt_adaptations"] = adaptations
        
        # Store detected paper domain if provided
        if agent_output.get("paper_domain"):
            result["paper_domain"] = agent_output["paper_domain"]
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Prompt adaptor LLM call failed: {e}. Using default prompts.")
        result["prompt_adaptations"] = []
    
    return result


def plan_node(state: ReproState) -> dict:
    """
    PlannerAgent: Analyze paper and create reproduction plan.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: This node makes LLM calls with full paper text, so it must 
    check context first. The planner receives the largest context of any node.
    """
    start_time = datetime.now(timezone.utc)
    logger = logging.getLogger(__name__)
    
    # Validate paper text exists and is non-empty
    paper_text = state.get("paper_text", "")
    if not paper_text or len(paper_text.strip()) < 100:
        logger.error(
            f"paper_text is missing or too short ({len(paper_text) if paper_text else 0} chars). "
            "Planner requires paper text to create reproduction plan."
        )
        return {
            "workflow_phase": "planning",
            "ask_user_trigger": "missing_paper_text",
            "pending_user_questions": [
                f"ERROR: Paper text is missing or too short ({len(paper_text) if paper_text else 0} characters). "
                "Planner requires paper text to create reproduction plan. "
                "Please provide paper text via paper_loader or check paper input."
            ],
            "awaiting_user_input": True,
        }
    
    # Context check - critical for planner
    escalation = check_context_or_escalate(state, "plan")
    if escalation is not None:
        if escalation.get("awaiting_user_input"):
            return escalation
        # Just state updates (e.g., metrics) - merge into state and continue
        state = {**state, **escalation}
    
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("planner", state)
    
    # Inject replan context for learning
    replan_count = state.get("replan_count", 0)
    if replan_count > 0:
        system_prompt += f"\n\nNOTE: This is Replan Attempt #{replan_count}. Previous plan was rejected. Improve strategy based on feedback."
    
    # Build user content with paper text and figures
    user_content = build_user_content_for_planner(state)
    
    # Call LLM with structured output
    try:
        agent_output = call_agent_with_metrics(
            agent_name="planner",
            system_prompt=system_prompt,
            user_content=user_content,
            state=state,
        )
    except Exception as e:
        logger.error(f"Planner LLM call failed: {e}")
        return create_llm_error_escalation("planner", "planning", e)
    
    # Extract results from agent output
    plan_data = {
        "paper_id": agent_output.get("paper_id", state.get("paper_id", "unknown")),
        "paper_domain": agent_output.get("paper_domain", "other"),
        "title": agent_output.get("title", ""),
        "summary": agent_output.get("summary", ""),
        "stages": agent_output.get("stages", []),
        "targets": agent_output.get("targets", []),
        "extracted_parameters": agent_output.get("extracted_parameters", []),
    }
    
    # Extract planned materials and assumptions
    planned_materials = agent_output.get("planned_materials", [])
    assumptions = agent_output.get("assumptions", {})
    
    result: Dict[str, Any] = {
        "workflow_phase": "planning",
        "plan": plan_data,
        "planned_materials": planned_materials,
        "assumptions": assumptions,
        "paper_domain": agent_output.get("paper_domain", "other"),
    }
    
    # Initialize progress stages from plan
    if result.get("plan") and result["plan"].get("stages"):
        state_with_plan = {**state, **result}
        # Force reset of progress on replan
        if "progress" in state_with_plan:
            state_with_plan["progress"] = None 
        
        try:
            state_with_plan = initialize_progress_from_plan(state_with_plan)
            state_with_plan = sync_extracted_parameters(state_with_plan)
            result["progress"] = state_with_plan.get("progress")
            result["extracted_parameters"] = state_with_plan.get("extracted_parameters")
        except Exception as e:
            logger.error(
                f"Failed to initialize progress from plan: {e}. "
                "This indicates a plan structure issue. Marking plan for revision."
            )
            # CRITICAL: Initialize default keys for rejection
            result["last_plan_review_verdict"] = "needs_revision"
            result["planner_feedback"] = (
                f"Progress initialization failed: {str(e)}. "
                "Please check plan structure and ensure all stages have required fields."
            )
            # Bounds check
            current_replan_count = state.get("replan_count", 0)
            runtime_config = state.get("runtime_config", {})
            max_replans = runtime_config.get("max_replans", MAX_REPLANS)
            if current_replan_count < max_replans:
                result["replan_count"] = current_replan_count + 1
            else:
                result["replan_count"] = current_replan_count
    
    # Log metrics
    log_agent_call("PlannerAgent", "plan", start_time)(state, result)
    
    return result


@with_context_check("plan_review")
def plan_reviewer_node(state: ReproState) -> dict:
    """
    PlanReviewerAgent: Review reproduction plan before stage execution.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT:
    - Sets `last_plan_review_verdict` state field.
    - Increments `replan_count` when verdict is "needs_revision".
    
    Note: Context check is handled by @with_context_check decorator.
    """
    logger = logging.getLogger(__name__)

    # State validation
    validation_issues = validate_state_or_warn(state, "plan_review")
    blocking_issues = [i for i in validation_issues if i.startswith("PLAN_ISSUE:")]
    
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("plan_reviewer", state)
    
    # Validate plan has stages
    plan = state.get("plan", {})
    plan_stages = plan.get("stages", [])
    
    if not plan_stages or len(plan_stages) == 0:
        blocking_issues.append(
            "PLAN_ISSUE: Plan must contain at least one stage. "
            "Current plan has no stages defined."
        )
    
    # Validate stages have targets
    for stage in plan_stages:
        stage_id = stage.get("stage_id", "unknown")
        targets = stage.get("targets", [])
        target_details = stage.get("target_details", [])
        
        has_targets = bool(targets) or bool(target_details)
        
        if not has_targets:
            blocking_issues.append(
                f"PLAN_ISSUE: Stage '{stage_id}' has no targets defined. "
                "Each stage must have at least one target figure to reproduce."
            )
    
    # Detect circular dependencies
    if plan_stages:
        stage_ids = {s.get("stage_id") for s in plan_stages if s.get("stage_id")}
        
        def detect_cycles() -> List[List[str]]:
            """Detect circular dependencies using DFS."""
            cycles: List[List[str]] = []
            visited: set = set()
            rec_stack: set = set()
            path: List[str] = []
            
            def dfs(stage_id: str):
                if stage_id in rec_stack:
                    cycle_start = path.index(stage_id)
                    cycle = path[cycle_start:] + [stage_id]
                    cycles.append(cycle)
                    return
                
                if stage_id in visited:
                    return
                
                visited.add(stage_id)
                rec_stack.add(stage_id)
                path.append(stage_id)
                
                stage = next((s for s in plan_stages if s.get("stage_id") == stage_id), None)
                if stage:
                    dependencies = stage.get("dependencies", [])
                    for dep_id in dependencies:
                        if dep_id in stage_ids:
                            dfs(dep_id)
                
                path.pop()
                rec_stack.remove(stage_id)
            
            for sid in stage_ids:
                if sid not in visited:
                    dfs(sid)
            
            return cycles
        
        cycles = detect_cycles()
        if cycles:
            cycle_descriptions = [
                " → ".join(cycle) + " (circular)"
                for cycle in cycles
            ]
            blocking_issues.append(
                f"PLAN_ISSUE: Circular dependencies detected: {', '.join(cycle_descriptions)}. "
                "Stages cannot depend on themselves or form dependency cycles. "
                "Please fix the dependency graph."
            )
        
        # Check self-dependencies
        for stage in plan_stages:
            stage_id = stage.get("stage_id")
            dependencies = stage.get("dependencies", [])
            if stage_id in dependencies:
                blocking_issues.append(
                    f"PLAN_ISSUE: Stage '{stage_id}' depends on itself. "
                    "Stages cannot depend on themselves."
                )
    
    # Handle blocking issues or call LLM
    if blocking_issues:
        agent_output = {
            "verdict": "needs_revision",
            "issues": [{"severity": "blocking", "description": issue} for issue in blocking_issues],
            "summary": f"Plan has {len(blocking_issues)} blocking issue(s) requiring revision",
            "feedback": "The following issues must be resolved:\n" + "\n".join(blocking_issues),
        }
    else:
        user_content = f"# REPRODUCTION PLAN TO REVIEW\n\n```json\n{json.dumps(plan, indent=2)}\n```"
        
        assumptions = state.get("assumptions", {})
        if assumptions:
            user_content += f"\n\n# ASSUMPTIONS\n\n```json\n{json.dumps(assumptions, indent=2)}\n```"
        
        try:
            agent_output = call_agent_with_metrics(
                agent_name="plan_reviewer",
                system_prompt=system_prompt,
                user_content=user_content,
                state=state,
            )
        except Exception as e:
            logger.error(f"Plan reviewer LLM call failed: {e}")
            agent_output = create_llm_error_auto_approve("plan_reviewer", e)
    
    result: Dict[str, Any] = {
        "workflow_phase": "plan_review",
        "last_plan_review_verdict": agent_output["verdict"],
    }
    
    # Increment replan counter if needs_revision
    if agent_output["verdict"] == "needs_revision":
        current_replan_count = state.get("replan_count", 0)
        runtime_config = state.get("runtime_config", {})
        max_replans = runtime_config.get("max_replans", MAX_REPLANS)
        if current_replan_count < max_replans:
            result["replan_count"] = current_replan_count + 1
        else:
            result["replan_count"] = current_replan_count
        result["planner_feedback"] = agent_output.get("feedback", agent_output.get("summary", ""))
    
    return result

