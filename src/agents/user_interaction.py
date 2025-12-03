"""
User interaction agent nodes: ask_user_node, material_checkpoint_node.

These nodes handle user interactions and mandatory checkpoints.

State Keys
----------
ask_user_node:
    READS: pending_user_questions, ask_user_trigger, paper_id, user_responses
    WRITES: workflow_phase, user_responses, pending_user_questions,
            awaiting_user_input, user_validation_attempts_*

material_checkpoint_node:
    READS: current_stage_id, stage_outputs, progress
    WRITES: workflow_phase, pending_validated_materials, pending_user_questions,
            awaiting_user_input, ask_user_trigger, last_node_before_ask_user
"""

import os
import signal
import logging
from typing import Dict, Any

from schemas.state import ReproState, save_checkpoint

from .helpers.context import validate_user_responses
from .helpers.materials import (
    extract_validated_materials,
    format_material_checkpoint_question,
)


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
    
    # Get original questions (for mapping responses after retry) or use current questions
    original_questions = state.get("original_user_questions", questions)
    # If this is a fresh call (not a retry), store original questions
    if "original_user_questions" not in state:
        original_questions = questions.copy()
    
    if not questions:
        return {
            "awaiting_user_input": False,
            "workflow_phase": "awaiting_user",
        }
    
    # Non-interactive mode
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
    
    # Interactive mode
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
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        else:
            print(f"NOTE: Timeout of {timeout_seconds}s disabled (Windows/non-Unix detected).")
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- Question {i}/{len(questions)} ---")
            print(f"\n{question}")
            print("-" * 40)
            
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
        
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        
        print("\n" + "=" * 60)
        print("✓ All responses collected")
        print("=" * 60)
            
    except KeyboardInterrupt:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            if 'old_handler' in locals():
                signal.signal(signal.SIGALRM, old_handler)
        print(f"\n\n⚠ Interrupted by user (Ctrl+C)")
        checkpoint_path = save_checkpoint(state, f"interrupted_{trigger}")
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        print("\nResume later with:")
        print(f"  python -m src.graph --resume {checkpoint_path}")
        raise SystemExit(0)
        
    except TimeoutError:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            if 'old_handler' in locals():
                signal.signal(signal.SIGALRM, old_handler)
        print(f"\n\n⚠ User response timeout ({timeout_seconds}s)")
        checkpoint_path = save_checkpoint(state, f"timeout_{trigger}")
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        print("\nResume later with:")
        print(f"  python -m src.graph --resume {checkpoint_path}")
        raise SystemExit(0)
        
    except EOFError:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            if 'old_handler' in locals():
                signal.signal(signal.SIGALRM, old_handler)
        print(f"\n\n⚠ End of input (EOF)")
        checkpoint_path = save_checkpoint(state, f"eof_{trigger}")
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        print("\nResume later with:")
        print(f"  python -m src.graph --resume {checkpoint_path}")
        raise SystemExit(0)
    
    # Map responses back to original questions if this is a retry
    # (responses might be keyed by reask question text)
    mapped_responses = {}
    for i, orig_q in enumerate(original_questions):
        # Try to find response by original question first
        if orig_q in responses:
            mapped_responses[orig_q] = responses[orig_q]
        # If not found, try by index (for reask scenarios where question text was modified)
        elif i < len(questions) and questions[i] in responses:
            mapped_responses[orig_q] = responses[questions[i]]
        # If still not found, check if any response key ends with the original question
        # (for reask format: "error message\n\nPlease try again:\n{original_question}")
        else:
            for resp_key, resp_value in responses.items():
                if resp_key.endswith(orig_q) or f"\n{orig_q}" in resp_key:
                    mapped_responses[orig_q] = resp_value
                    break
    
    # Fallback: if mapping failed but we have responses, use them as-is
    # (this handles edge cases where mapping logic doesn't catch everything)
    if not mapped_responses and responses:
        # If we have the same number of responses as original questions, map by index
        if len(responses) == len(original_questions):
            response_values = list(responses.values())
            mapped_responses = {orig_q: response_values[i] for i, orig_q in enumerate(original_questions)}
        else:
            # Last resort: use responses as-is (might cause validation issues, but better than empty)
            mapped_responses = responses
    
    # Validate user responses using original questions
    validation_errors = validate_user_responses(trigger, mapped_responses, original_questions)
    if validation_errors:
        validation_attempt_key = f"user_validation_attempts_{trigger}"
        validation_attempts = state.get(validation_attempt_key, 0) + 1
        max_validation_attempts = 3
        
        if validation_attempts >= max_validation_attempts:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"User validation failed {validation_attempts} times for trigger '{trigger}'. "
                "Accepting response despite validation errors and escalating to supervisor."
            )
            return {
                "user_responses": {**state.get("user_responses", {}), **mapped_responses},
                "pending_user_questions": [],
                "awaiting_user_input": False,
                "workflow_phase": "awaiting_user",
                validation_attempt_key: 0,
                "original_user_questions": None,  # Clear after success
                "supervisor_feedback": (
                    f"User response had validation errors but was accepted after {validation_attempts} attempts."
                ),
            }
        
        error_msg = "\n".join(f"  - {err}" for err in validation_errors)
        
        # Prepend error message to the first question, and keep all other questions
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
            "original_user_questions": original_questions,  # Preserve for mapping responses
        }
    
    validation_attempt_key = f"user_validation_attempts_{trigger}"
    result = {
        "user_responses": {**state.get("user_responses", {}), **mapped_responses},
        "pending_user_questions": [],
        "awaiting_user_input": False,
        "workflow_phase": "awaiting_user",
        validation_attempt_key: 0,
        "original_user_questions": None,  # Clear after successful validation
    }
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

