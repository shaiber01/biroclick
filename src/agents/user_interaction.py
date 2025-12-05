"""
User interaction agent nodes: ask_user_node, material_checkpoint_node.

These nodes handle user interactions and mandatory checkpoints.

State Keys
----------
ask_user_node:
    READS: pending_user_questions, ask_user_trigger, paper_id
    WRITES: workflow_phase, user_responses, pending_user_questions,
            awaiting_user_input, user_validation_attempts_*

material_checkpoint_node:
    READS: current_stage_id, stage_outputs, progress
    WRITES: workflow_phase, pending_validated_materials, pending_user_questions,
            awaiting_user_input, ask_user_trigger, last_node_before_ask_user

Implementation Notes
--------------------
This module uses LangGraph's `interrupt()` function for human-in-the-loop workflows.
When ask_user_node needs user input, it calls `interrupt(payload)` which:
1. Pauses the graph execution
2. Returns the payload to the caller (runner.py)
3. When resumed with `Command(resume=user_response)`, the interrupt() call returns user_response
4. The node continues execution with the user's response

This is cleaner than the interrupt_before pattern because:
- The node runs only once (not twice)
- No "pre-provided response" handling needed
- User response flows directly into the node via interrupt()'s return value
"""

import os
import signal
import logging
from typing import Dict, Any

from langgraph.types import interrupt

from schemas.state import ReproState, save_checkpoint

from .helpers.context import validate_user_responses
from .helpers.materials import (
    extract_validated_materials,
    format_material_checkpoint_question,
)


def _format_boxed_content(title: str, content: str, prefix: str = "   ") -> str:
    """Format content in a box for clear log visibility."""
    lines = content.strip().split('\n')
    boxed_lines = [f"{prefix}┌─── {title} ───"]
    for line in lines:
        boxed_lines.append(f"{prefix}│ {line}")
    boxed_lines.append(f"{prefix}└{'─' * (len(title) + 6)}")
    return '\n'.join(boxed_lines)


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
            "awaiting_user_input": False,
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
    
    # Get original questions (for mapping responses after retry) or use current questions
    original_questions = state.get("original_user_questions", questions)
    # If this is a fresh call (not a retry), store original questions
    if "original_user_questions" not in state:
        original_questions = questions.copy()
    
    # Log the questions being asked
    logger.info(f"❓ ask_user: trigger={trigger}, {len(questions)} question(s)")
    for i, q in enumerate(questions, 1):
        logger.info(f"\n{_format_boxed_content(f'QUESTION {i}', q)}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # INTERRUPT: Pause execution and wait for user response
    # The interrupt() call pauses the graph. When resumed with 
    # Command(resume=user_response), interrupt() returns that value.
    # ═══════════════════════════════════════════════════════════════════════
    user_response = interrupt({
        "trigger": trigger,
        "questions": questions,
        "paper_id": paper_id,
    })
    
    # ═══════════════════════════════════════════════════════════════════════
    # RESUMED: User has provided a response
    # ═══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{_format_boxed_content('USER RESPONSE', str(user_response))}")
    
    # Map the response to the first question (typically there's just one question)
    mapped_responses = {}
    if original_questions:
        mapped_responses[original_questions[0]] = user_response
    
    # Validate user response
    validation_errors = validate_user_responses(trigger, mapped_responses, original_questions)
    if validation_errors:
        validation_attempt_key = f"user_validation_attempts_{trigger}"
        validation_attempts = state.get(validation_attempt_key, 0) + 1
        max_validation_attempts = 3
        
        if validation_attempts >= max_validation_attempts:
            logger.warning(
                f"User validation failed {validation_attempts} times for trigger '{trigger}'. "
                "Accepting response despite validation errors and escalating to supervisor."
            )
            result = {
                "user_responses": {**state.get("user_responses", {}), **mapped_responses},
                "pending_user_questions": [],
                "awaiting_user_input": False,
                "workflow_phase": "awaiting_user",
                validation_attempt_key: 0,
                "original_user_questions": None,
                "supervisor_feedback": (
                    f"User response had validation errors but was accepted after {validation_attempts} attempts."
                ),
            }
            if safety_net_triggered:
                result["ask_user_trigger"] = trigger
            return result
        
        error_msg = "\n".join(f"  - {err}" for err in validation_errors)
        first_question = original_questions[0] if original_questions else 'Please provide a valid response.'
        reask_questions = [
            f"Your response had validation errors (attempt {validation_attempts}/{max_validation_attempts}):\n{error_msg}\n\n"
            f"Please try again:\n{first_question}"
        ]
        
        if len(original_questions) > 1:
            reask_questions.extend(original_questions[1:])
        
        return {
            "pending_user_questions": reask_questions,
            "awaiting_user_input": True,
            "ask_user_trigger": trigger,
            "last_node_before_ask_user": state.get("last_node_before_ask_user"),
            validation_attempt_key: validation_attempts,
            "original_user_questions": original_questions,
        }
    
    # Validation passed - return success
    validation_attempt_key = f"user_validation_attempts_{trigger}"
    result = {
        "user_responses": {**state.get("user_responses", {}), **mapped_responses},
        "pending_user_questions": [],
        "awaiting_user_input": False,
        "workflow_phase": "awaiting_user",
        validation_attempt_key: 0,
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
            "awaiting_user_input": False,
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
        "awaiting_user_input": True,
        "ask_user_trigger": "material_checkpoint",
        "last_node_before_ask_user": "material_checkpoint",
        "pending_validated_materials": pending_materials,
    }

