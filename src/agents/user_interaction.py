"""
User interaction agent nodes: ask_user_node, material_checkpoint_node.

These nodes handle user interactions and mandatory checkpoints.

State Keys
----------
ask_user_node:
    READS: pending_user_questions, ask_user_trigger, paper_id
    WRITES: workflow_phase, user_responses, pending_user_questions

material_checkpoint_node:
    READS: current_stage_id, stage_outputs, progress
    WRITES: workflow_phase, pending_validated_materials, pending_user_questions,
            ask_user_trigger, last_node_before_ask_user

Implementation Notes
--------------------
This module uses LangGraph's `interrupt()` function for human-in-the-loop workflows.
When ask_user_node needs user input, it calls `interrupt(payload)` which:
1. Pauses the graph execution
2. Returns the payload to the caller (runner.py)
3. When resumed with `Command(resume=user_response)`, the interrupt() call returns user_response
4. The node continues execution with the user's response

IMPORTANT: When resumed, the entire node function re-executes from the beginning.
The interrupt() call returns the user's response on the second execution instead
of pausing. Therefore:
- Code BEFORE interrupt() runs TWICE (keep it side-effect free)
- Code AFTER interrupt() runs ONCE (put logging/state changes here)
"""

import os
import signal
import logging
from typing import Dict, Any

from langgraph.types import interrupt

from schemas.state import ReproState, save_checkpoint

from .helpers.materials import (
    extract_validated_materials,
    format_material_checkpoint_question,
)
from .user_options import get_options_prompt


def _format_boxed_content(title: str, content: str, prefix: str = "   ") -> str:
    """Format content in a box for clear log visibility."""
    lines = content.strip().split('\n')
    boxed_lines = [f"{prefix}┌─── {title} ───"]
    for line in lines:
        boxed_lines.append(f"{prefix}│ {line}")
    boxed_lines.append(f"{prefix}└{'─' * (len(title) + 6)}")
    return '\n'.join(boxed_lines)


def _infer_error_context(state: "ReproState") -> str:
    """
    Infer what went wrong by checking which verdict fields are None/invalid.
    
    When ask_user is called without a trigger (safety net), this helps
    generate a more contextual error message by determining which
    validation step likely failed.
    
    Returns a context string like "physics_error", "execution_error", etc.
    """
    # Check verdict fields in order of the pipeline
    # A None verdict typically means the node was skipped or failed silently
    if state.get("physics_verdict") is None and state.get("execution_verdict") is not None:
        return "physics_error"
    if state.get("execution_verdict") is None:
        return "execution_error"
    if state.get("comparison_verdict") is None:
        return "comparison_error"
    if state.get("last_code_review_verdict") is None:
        return "code_review_error"
    if state.get("last_design_review_verdict") is None:
        return "design_review_error"
    if state.get("last_plan_review_verdict") is None:
        return "plan_review_error"
    
    # Check if ask_user_trigger is stuck (indicates previous interaction wasn't completed)
    if state.get("ask_user_trigger"):
        return "stuck_awaiting_input"
    
    return "unknown_error"


def _generate_error_question(context: str, state: "ReproState") -> str:
    """
    Generate an appropriate error question based on inferred context.
    
    Provides a more helpful message to the user than a generic
    "unexpected error" when we can determine what likely went wrong.
    """
    stage_id = state.get("current_stage_id", "unknown")
    
    error_messages = {
        "physics_error": (
            f"Physics validation failed to run for stage '{stage_id}'.\n\n"
            "This may indicate the validation node was skipped or encountered an error.\n"
            "The simulation may have completed but physics checks were not performed."
        ),
        "execution_error": (
            f"Execution validation failed to run for stage '{stage_id}'.\n\n"
            "The simulation may not have completed properly, or the validation was skipped."
        ),
        "comparison_error": (
            f"Comparison check failed to run for stage '{stage_id}'.\n\n"
            "Results analysis may not have completed."
        ),
        "code_review_error": (
            f"Code review failed to run for stage '{stage_id}'.\n\n"
            "The generated code may not have been reviewed."
        ),
        "design_review_error": (
            f"Design review failed to run for stage '{stage_id}'.\n\n"
            "The simulation design may not have been validated."
        ),
        "plan_review_error": (
            "Plan review failed to run.\n\n"
            "The reproduction plan may not have been validated."
        ),
        "stuck_awaiting_input": (
            "Workflow appears to have an unprocessed ask_user_trigger.\n\n"
            "This indicates a previous user interaction wasn't properly completed.\n"
            "The system will attempt to recover."
        ),
        "unknown_error": (
            "An unexpected workflow error occurred.\n\n"
            "Unable to determine the specific cause from the current state."
        ),
    }
    
    message = error_messages.get(context, error_messages["unknown_error"])
    return f"WORKFLOW RECOVERY NEEDED\n\n{message}"


def ask_user_node(state: ReproState) -> Dict[str, Any]:
    """
    User interaction node using LangGraph's interrupt() for human-in-the-loop.
    
    This node uses the interrupt() function to pause execution and get user input.
    When the graph is resumed with Command(resume=user_response), the interrupt()
    call returns the user's response directly.
    
    Flow:
    1. Check if there are questions to ask
    2. Call interrupt() with the questions and trigger info
    3. Graph pauses, runner.py displays questions and gets user input
    4. Runner resumes with Command(resume=user_response)
    5. interrupt() returns the user response
    6. Node validates and returns state update
        
    Returns:
        Dict with state updates (user_responses, cleared pending questions)
    """
    logger = logging.getLogger(__name__)
    
    questions = state.get("pending_user_questions", [])
    trigger = state.get("ask_user_trigger")
    paper_id = state.get("paper_id", "unknown")
    
    # Early return if nothing to ask - this is the normal "no questions" case
    if not questions:
        return {
            "workflow_phase": "awaiting_user",
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAFETY NET: Ensure we always have a trigger when we have questions
    # ═══════════════════════════════════════════════════════════════════════
    safety_net_triggered = False
    if not trigger:
        logger.warning(
            "ask_user_node called with questions but without ask_user_trigger - "
            "this indicates a workflow bug. Setting generic 'unknown_escalation' trigger."
        )
        trigger = "unknown_escalation"
        safety_net_triggered = True
        
        # BUG FIX: When safety net triggers, the existing questions in state may have
        # been generated for a different trigger with different valid options.
        # Use contextual helpers to infer what went wrong and generate appropriate questions.
        error_context = _infer_error_context(state)
        contextual_message = _generate_error_question(error_context, state)
        
        # Include original context if available (but strip old Options section)
        original_context = questions[0] if questions else ""
        if original_context and "Options:" in original_context:
            original_context = original_context.split("Options:")[0].strip()
        
        # Build the final question with contextual error info and correct options
        if original_context:
            questions = [
                f"{contextual_message}\n\n"
                f"Original context:\n{original_context}\n\n"
                f"{get_options_prompt('unknown_escalation')}"
            ]
        else:
            questions = [
                f"{contextual_message}\n\n"
                f"{get_options_prompt('unknown_escalation')}"
            ]
        logger.info(f"Safety net: inferred error context '{error_context}', regenerated questions")
    
    # Get original questions (for mapping responses after retry) or use current questions.
    # NOTE: We use `or questions` instead of the default parameter because
    # state.get("key", default) returns None if key exists with value None,
    # not the default. Previous successful ask_user calls set this to None,
    # which would poison subsequent calls without this fix.
    original_questions = state.get("original_user_questions") or questions
    # If this is a fresh call (not a retry), make a copy to avoid mutation
    if not state.get("original_user_questions"):
        original_questions = questions.copy() if questions else []
    
    # ═══════════════════════════════════════════════════════════════════════
    # INTERRUPT: Pause execution and wait for user response
    # The interrupt() call pauses the graph. When resumed with 
    # Command(resume=user_response), interrupt() returns that value.
    # NOTE: Code BEFORE this point runs TWICE (on initial call and on resume).
    #       Code AFTER this point runs ONCE (only after user responds).
    # ═══════════════════════════════════════════════════════════════════════
    user_response = interrupt({
        "trigger": trigger,
        "questions": questions,
        "paper_id": paper_id,
    })
    
    # ═══════════════════════════════════════════════════════════════════════
    # RESUMED: User has provided a response (this code runs only once)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info(f"❓ ask_user: trigger={trigger}, {len(questions)} question(s), received response")
    logger.info(f"\n{_format_boxed_content('USER RESPONSE', str(user_response))}")
    
    # Map the response to the first question (typically there's just one question)
    mapped_responses = {}
    if original_questions:
        mapped_responses[original_questions[0]] = user_response
    
    # ═══════════════════════════════════════════════════════════════════════
    # SIMPLIFIED VALIDATION: Only check for empty responses.
    # 
    # Keyword validation is NOT done here because:
    # 1. Supervisor is the ONLY node that follows ask_user (route_after_ask_user always returns "supervisor")
    # 2. Every trigger handler in supervisor already has validation logic
    # 3. Some handlers (like reviewer_escalation) use LLM fallback for free-form responses
    # 4. Having validation here caused bugs where responses were not properly stored
    #
    # The supervisor will validate the response and route back to ask_user if needed.
    # ═══════════════════════════════════════════════════════════════════════
    if not str(user_response).strip():
        logger.warning(f"Empty response received for trigger '{trigger}', asking user to retry")
        first_question = original_questions[0] if original_questions else 'Please provide a response.'
        return {
            "pending_user_questions": [
                f"Your response was empty. Please provide a response:\n\n{first_question}"
            ],
            "ask_user_trigger": trigger,
            "last_node_before_ask_user": state.get("last_node_before_ask_user"),
            "original_user_questions": original_questions,
        }
    
    # Store response and pass to supervisor for validation/routing
    result = {
        "user_responses": {**(state.get("user_responses") or {}), **mapped_responses},
        "pending_user_questions": [],
        "workflow_phase": "awaiting_user",
        "original_user_questions": None,
    }
    if safety_net_triggered:
        result["ask_user_trigger"] = trigger
    return result


def material_checkpoint_node(state: ReproState) -> dict:
    """
    Mandatory material validation checkpoint after Stage 0.
    
    This node ALWAYS routes to ask_user to require user confirmation
    of material validation results before proceeding to Stage 1+.
    
    IMPORTANT: This node stores extracted materials in `pending_validated_materials`,
    NOT in `validated_materials`. The supervisor_node will move them to 
    `validated_materials` ONLY when the user approves.
    
    Per global_rules.md RULE 0A:
    "After Stage 0 completes, you MUST pause and ask the user to confirm
    the material optical constants are correct before proceeding."
    
    Returns:
        Dict with state updates including pending_user_questions and pending_validated_materials
    """
    logger = logging.getLogger(__name__)
    
    # ═══════════════════════════════════════════════════════════════════════
    # SKIP CHECK: If materials are already validated, don't ask again
    # This prevents infinite loops when route_after_supervisor routes here
    # after user has already approved materials.
    # ═══════════════════════════════════════════════════════════════════════
    validated_materials = state.get("validated_materials", [])
    if validated_materials:
        logger.info(
            f"material_checkpoint: Materials already validated ({len(validated_materials)} items), skipping checkpoint"
        )
        # Return empty update to pass through to next edge (which goes to ask_user,
        # but ask_user will see no pending_user_questions and pass through quickly)
        return {
            "workflow_phase": "material_checkpoint",
            "pending_user_questions": [],  # No questions = ask_user passes through
        }
    
    # Get material validation results from progress
    progress = state.get("progress", {})
    stages = progress.get("stages", [])
    
    # Find Stage 0 (material validation) results
    stage0_info = None
    # Handle None or non-iterable stages gracefully
    if stages is not None and isinstance(stages, (list, tuple)):
        for stage in stages:
            # Only process dict-like objects
            if isinstance(stage, dict) and stage.get("stage_type") == "MATERIAL_VALIDATION":
                stage0_info = stage
                break
    
    # Get output files from stage_outputs
    stage_outputs = state.get("stage_outputs", {})
    output_files = stage_outputs.get("files", [])
    # Handle None or non-iterable files gracefully, use case-insensitive matching
    plot_files = []
    if output_files is not None and isinstance(output_files, (list, tuple)):
        plot_files = [
            f for f in output_files 
            if isinstance(f, str) and f.lower().endswith(('.png', '.pdf', '.jpg'))
        ]
    
    # Extract materials - stored as PENDING until user approves
    pending_materials = extract_validated_materials(state)
    
    # Guard against empty materials
    warning_msg = ""
    if not pending_materials:
        warning_msg = (
            "\n\n⚠️ WARNING: No materials were automatically detected! "
            "Code generation will FAIL without materials. "
            "Please select 'CHANGE_MATERIAL' or 'CHANGE_DATABASE' to specify them manually."
        )
    
    # Build the checkpoint question per global_rules.md RULE 0A format
    question = format_material_checkpoint_question(state, stage0_info, plot_files, pending_materials)
    question += warning_msg
    
    return {
        "workflow_phase": "material_checkpoint",
        "pending_user_questions": [question],
        "ask_user_trigger": "material_checkpoint",
        "last_node_before_ask_user": "material_checkpoint",
        "pending_validated_materials": pending_materials,
    }

