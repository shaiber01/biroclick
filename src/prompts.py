"""
Prompt Builder for ReproLab Multi-Agent System

This module handles:
1. Loading base prompts from prompts/*.md files
2. Substituting placeholders like {THRESHOLDS_TABLE}
3. Prepending global_rules.md to each agent's prompt
4. Applying prompt adaptations from PromptAdaptorAgent

═══════════════════════════════════════════════════════════════════════════════
IMPORTANT: This is the ONLY module that should build prompts for agents.
All agent implementations should call build_agent_prompt() to get their
final system prompt with all substitutions and global rules applied.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from schemas.state import format_thresholds_table, ReproState


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# Default prompts directory (relative to project root)
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Agent prompt filenames (without .md extension)
AGENT_PROMPTS = {
    "prompt_adaptor": "prompt_adaptor_agent",
    "planner": "planner_agent",
    "plan_reviewer": "plan_reviewer_agent",
    "simulation_designer": "simulation_designer_agent",
    "design_reviewer": "design_reviewer_agent",
    "code_generator": "code_generator_agent",
    "code_reviewer": "code_reviewer_agent",
    "execution_validator": "execution_validator_agent",
    "physics_sanity": "physics_sanity_agent",
    "results_analyzer": "results_analyzer_agent",
    "comparison_validator": "comparison_validator_agent",
    "supervisor": "supervisor_agent",
}

# Placeholder tokens and their generators
PLACEHOLDER_GENERATORS = {
    "{THRESHOLDS_TABLE}": format_thresholds_table,
}


# ═══════════════════════════════════════════════════════════════════════
# Prompt Loading
# ═══════════════════════════════════════════════════════════════════════

def load_prompt_file(filename: str, prompts_dir: Optional[Path] = None) -> str:
    """
    Load a prompt file from the prompts directory.
    
    Args:
        filename: Prompt filename (with or without .md extension)
        prompts_dir: Optional override for prompts directory
        
    Returns:
        Prompt file contents as string
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    if prompts_dir is None:
        prompts_dir = PROMPTS_DIR
    
    # Add .md extension if not present
    if not filename.endswith(".md"):
        filename = f"{filename}.md"
    
    filepath = prompts_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_global_rules(prompts_dir: Optional[Path] = None) -> str:
    """
    Load the global_rules.md file.
    
    Args:
        prompts_dir: Optional override for prompts directory
        
    Returns:
        Global rules prompt content
    """
    return load_prompt_file("global_rules", prompts_dir)


# ═══════════════════════════════════════════════════════════════════════
# Placeholder Substitution
# ═══════════════════════════════════════════════════════════════════════

def substitute_placeholders(prompt: str) -> str:
    """
    Substitute all known placeholders in a prompt with their generated values.
    
    Currently handles:
    - {THRESHOLDS_TABLE}: Discrepancy thresholds table from state.py
    
    Args:
        prompt: Prompt string with potential placeholders
        
    Returns:
        Prompt with all placeholders substituted
        
    Example:
        >>> prompt = "Use these thresholds:\n{THRESHOLDS_TABLE}"
        >>> result = substitute_placeholders(prompt)
        >>> "{THRESHOLDS_TABLE}" in result
        False
    """
    result = prompt
    
    for placeholder, generator in PLACEHOLDER_GENERATORS.items():
        if placeholder in result:
            # Call the generator function to get the replacement value
            replacement = generator()
            result = result.replace(placeholder, replacement)
    
    return result


# ═══════════════════════════════════════════════════════════════════════
# Prompt Adaptation
# ═══════════════════════════════════════════════════════════════════════

def apply_prompt_adaptations(
    prompt: str,
    agent_name: str,
    adaptations: List[Dict[str, Any]]
) -> str:
    """
    Apply prompt adaptations from PromptAdaptorAgent.
    
    Adaptations are modifications specific to the current paper that adjust
    agent prompts for domain-specific needs.
    
    Args:
        prompt: Base prompt string
        agent_name: Name of the agent (e.g., "simulation_designer")
        adaptations: List of adaptation dicts from state["prompt_adaptations"]
        
    Returns:
        Modified prompt with adaptations applied
        
    Adaptation dict format:
        {
            "target_agent": "SimulationDesignerAgent",
            "modification_type": "append" | "prepend" | "replace" | "disable",
            "content": "... additional content ...",
            "section_marker": "optional marker for replace/disable",
            "confidence": 0.85,
            "reason": "why this adaptation is needed"
        }
    """
    if not adaptations:
        return prompt
    
    # Normalize agent name for matching
    agent_name_normalized = agent_name.lower().replace("_", "")
    
    result = prompt
    
    for adaptation in adaptations:
        # Check if this adaptation applies to this agent
        target = adaptation.get("target_agent", "")
        target_normalized = target.lower().replace("_", "").replace("agent", "")
        
        if agent_name_normalized not in target_normalized and target_normalized not in agent_name_normalized:
            continue
        
        mod_type = adaptation.get("modification_type", "")
        content = adaptation.get("content", "")
        
        if mod_type == "append":
            # Add content at the end
            result = f"{result}\n\n# Paper-Specific Adaptation\n{content}"
            
        elif mod_type == "prepend":
            # Add content at the beginning (after global rules if present)
            result = f"# Paper-Specific Adaptation\n{content}\n\n{result}"
            
        elif mod_type == "replace":
            # Replace a specific section (identified by marker)
            marker = adaptation.get("section_marker", "")
            if marker and marker in result:
                result = result.replace(marker, content)
                
        elif mod_type == "disable":
            # Comment out a section
            marker = adaptation.get("section_marker", "")
            if marker and marker in result:
                # Find and comment out the section
                result = result.replace(marker, f"[DISABLED: {marker}]")
    
    return result


# ═══════════════════════════════════════════════════════════════════════
# Main Prompt Builder
# ═══════════════════════════════════════════════════════════════════════

def build_agent_prompt(
    agent_name: str,
    state: Optional[ReproState] = None,
    include_global_rules: bool = True,
    prompts_dir: Optional[Path] = None
) -> str:
    """
    Build the complete system prompt for an agent.
    
    This function:
    1. Loads the agent's base prompt from prompts/{agent_name}_agent.md
    2. Loads and prepends global_rules.md (if include_global_rules=True)
    3. Substitutes placeholders like {THRESHOLDS_TABLE}
    4. Applies any prompt adaptations from state["prompt_adaptations"]
    
    Args:
        agent_name: Name of the agent (e.g., "simulation_designer", "code_generator")
        state: Optional ReproState for prompt adaptations (can be None)
        include_global_rules: Whether to prepend global rules (default: True)
        prompts_dir: Optional override for prompts directory
        
    Returns:
        Complete system prompt ready for LLM call
        
    Example:
        >>> prompt = build_agent_prompt("code_generator", state)
        >>> # prompt now contains:
        >>> # 1. global_rules.md content (with THRESHOLDS_TABLE substituted)
        >>> # 2. code_generator_agent.md content
        >>> # 3. Any paper-specific adaptations from state
    """
    # Determine prompt filename
    if agent_name in AGENT_PROMPTS:
        prompt_filename = AGENT_PROMPTS[agent_name]
    else:
        # Assume agent_name is already the filename stem
        prompt_filename = f"{agent_name}_agent"
    
    # Load the agent's base prompt
    agent_prompt = load_prompt_file(prompt_filename, prompts_dir)
    
    # Load and prepend global rules if requested
    if include_global_rules:
        global_rules = load_global_rules(prompts_dir)
        # Substitute placeholders in global rules
        global_rules = substitute_placeholders(global_rules)
        # Combine: global rules first, then agent-specific prompt
        combined_prompt = f"{global_rules}\n\n{'═' * 75}\n\n{agent_prompt}"
    else:
        combined_prompt = agent_prompt
    
    # Substitute any placeholders in the combined prompt
    combined_prompt = substitute_placeholders(combined_prompt)
    
    # Apply prompt adaptations if state is provided
    if state is not None:
        adaptations = state.get("prompt_adaptations", [])
        if adaptations:
            combined_prompt = apply_prompt_adaptations(
                combined_prompt, 
                agent_name, 
                adaptations
            )
    
    return combined_prompt


def get_agent_prompt_cached(
    agent_name: str,
    state: Optional[ReproState] = None,
    cache: Optional[Dict[str, str]] = None
) -> str:
    """
    Get agent prompt with optional caching for repeated calls.
    
    Args:
        agent_name: Name of the agent
        state: Optional ReproState for adaptations
        cache: Optional dict to use as cache
        
    Returns:
        Complete system prompt
    """
    # If no adaptations and cache exists, use cached version
    adaptations = state.get("prompt_adaptations", []) if state else []
    
    if cache is not None and not adaptations:
        if agent_name in cache:
            return cache[agent_name]
        
        prompt = build_agent_prompt(agent_name, state)
        cache[agent_name] = prompt
        return prompt
    
    return build_agent_prompt(agent_name, state)


# ═══════════════════════════════════════════════════════════════════════
# Validation Helpers
# ═══════════════════════════════════════════════════════════════════════

def validate_all_prompts_loadable(prompts_dir: Optional[Path] = None) -> Dict[str, bool]:
    """
    Validate that all agent prompts can be loaded.
    
    Useful for testing and CI to ensure prompt files exist.
    
    Args:
        prompts_dir: Optional override for prompts directory
        
    Returns:
        Dict mapping agent names to load success (True/False)
    """
    results = {}
    
    # Check global rules
    try:
        load_global_rules(prompts_dir)
        results["global_rules"] = True
    except FileNotFoundError:
        results["global_rules"] = False
    
    # Check each agent prompt
    for agent_name, prompt_filename in AGENT_PROMPTS.items():
        try:
            load_prompt_file(prompt_filename, prompts_dir)
            results[agent_name] = True
        except FileNotFoundError:
            results[agent_name] = False
    
    return results


def validate_placeholders_substituted(prompt: str) -> List[str]:
    """
    Check for any remaining unsubstituted placeholders in a prompt.
    
    Args:
        prompt: Prompt string to check
        
    Returns:
        List of unsubstituted placeholder tokens found
    """
    unsubstituted = []
    
    for placeholder in PLACEHOLDER_GENERATORS.keys():
        if placeholder in prompt:
            unsubstituted.append(placeholder)
    
    return unsubstituted


# ═══════════════════════════════════════════════════════════════════════
# ASK_USER Trigger Documentation
# ═══════════════════════════════════════════════════════════════════════
#
# This section documents all valid ask_user_trigger values for the system.
# The SupervisorAgent must handle each of these triggers appropriately.
#
# See: supervisor_agent.md and agents.py:supervisor_node
#
# ═══════════════════════════════════════════════════════════════════════

ASK_USER_TRIGGERS = {
    "material_checkpoint": {
        "description": "Mandatory Stage 0 material validation requires user confirmation",
        "source_node": "material_checkpoint",
        "expected_response_keys": ["verdict", "notes"],
        "valid_verdicts": ["APPROVE", "CHANGE_DATABASE", "CHANGE_MATERIAL", "NEED_HELP"],
        "supervisor_action": {
            "APPROVE": "Set supervisor_verdict='ok_continue', proceed to select_stage",
            "CHANGE_DATABASE": "Invalidate Stage 0, update assumptions, rerun Stage 0",
            "CHANGE_MATERIAL": "Route to plan with supervisor_feedback about material change",
            "NEED_HELP": "Route back to ask_user with additional context",
        }
    },
    "code_review_limit": {
        "description": "Code review revision limit (MAX_CODE_REVISIONS) exceeded",
        "source_node": "code_review",
        "expected_response_keys": ["action", "hint"],
        "valid_verdicts": ["PROVIDE_HINT", "SKIP_STAGE", "STOP"],
        "supervisor_action": {
            "PROVIDE_HINT": "Reset code_revision_count=0, add hint to reviewer_feedback, route to generate_code",
            "SKIP_STAGE": "Mark stage as blocked, route to select_stage",
            "STOP": "Route to generate_report",
        }
    },
    "design_review_limit": {
        "description": "Design review revision limit (MAX_DESIGN_REVISIONS) exceeded",
        "source_node": "design_review",
        "expected_response_keys": ["action", "hint"],
        "valid_verdicts": ["PROVIDE_HINT", "SKIP_STAGE", "STOP"],
        "supervisor_action": {
            "PROVIDE_HINT": "Reset design_revision_count=0, add hint to reviewer_feedback, route to design",
            "SKIP_STAGE": "Mark stage as blocked, route to select_stage",
            "STOP": "Route to generate_report",
        }
    },
    "execution_failure_limit": {
        "description": "Execution failure limit (MAX_EXECUTION_FAILURES) exceeded",
        "source_node": "execution_check",
        "expected_response_keys": ["action", "guidance"],
        "valid_verdicts": ["RETRY_WITH_GUIDANCE", "SKIP_STAGE", "STOP"],
        "supervisor_action": {
            "RETRY_WITH_GUIDANCE": "Reset execution_failure_count=0, add guidance to supervisor_feedback, route to generate_code",
            "SKIP_STAGE": "Mark stage as blocked, route to select_stage",
            "STOP": "Route to generate_report",
        }
    },
    "physics_failure_limit": {
        "description": "Physics sanity check failure limit (MAX_PHYSICS_FAILURES) exceeded",
        "source_node": "physics_check",
        "expected_response_keys": ["action", "guidance"],
        "valid_verdicts": ["RETRY_WITH_GUIDANCE", "ACCEPT_PARTIAL", "SKIP_STAGE", "STOP"],
        "supervisor_action": {
            "RETRY_WITH_GUIDANCE": "Reset physics_failure_count=0, add guidance, route to generate_code or design",
            "ACCEPT_PARTIAL": "Mark stage completed_partial, proceed to analyze",
            "SKIP_STAGE": "Mark stage as blocked, route to select_stage",
            "STOP": "Route to generate_report",
        }
    },
    "context_overflow": {
        "description": "LLM context window would overflow, recovery options needed",
        "source_node": "any (detected by check_context_before_node)",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["SUMMARIZE_FEEDBACK", "TRUNCATE_PAPER", "SKIP_STAGE", "STOP"],
        "supervisor_action": {
            "SUMMARIZE_FEEDBACK": "Apply summarize_feedback recovery action",
            "TRUNCATE_PAPER": "Apply truncate_paper_to_methods recovery action",
            "SKIP_STAGE": "Mark stage as blocked, route to select_stage",
            "STOP": "Route to generate_report",
        }
    },
    "replan_limit": {
        "description": "Replan limit (MAX_REPLANS) exceeded",
        "source_node": "plan_review",
        "expected_response_keys": ["action", "guidance"],
        "valid_verdicts": ["FORCE_ACCEPT", "PROVIDE_GUIDANCE", "STOP"],
        "supervisor_action": {
            "FORCE_ACCEPT": "Accept plan as-is, route to select_stage",
            "PROVIDE_GUIDANCE": "Reset replan_count=0, add guidance, route to plan",
            "STOP": "Route to generate_report",
        }
    },
    "backtrack_approval": {
        "description": "Backtrack suggestion requires user confirmation",
        "source_node": "supervisor",
        "expected_response_keys": ["approve", "alternative"],
        "valid_verdicts": ["APPROVE_BACKTRACK", "REJECT_BACKTRACK", "ALTERNATIVE"],
        "supervisor_action": {
            "APPROVE_BACKTRACK": "Route to handle_backtrack",
            "REJECT_BACKTRACK": "Clear backtrack_suggestion, continue normally",
            "ALTERNATIVE": "Apply user's alternative suggestion",
        }
    },
    "clarification": {
        "description": "Ambiguous paper information requires user clarification",
        "source_node": "any planning/design agent",
        "expected_response_keys": ["clarification"],
        "valid_verdicts": None,  # Free-form response
        "supervisor_action": "Add clarification to assumptions, continue from last_node_before_ask_user",
    },
    "unknown": {
        "description": "Unknown trigger - fallback handling",
        "source_node": "unknown",
        "expected_response_keys": ["action"],
        "valid_verdicts": ["CONTINUE", "STOP"],
        "supervisor_action": {
            "CONTINUE": "Route to select_stage",
            "STOP": "Route to generate_report",
        }
    },
}


def get_ask_user_trigger_info(trigger: str) -> Dict[str, Any]:
    """
    Get documentation for an ask_user trigger.
    
    Args:
        trigger: The ask_user_trigger value
        
    Returns:
        Dict with trigger documentation (or unknown trigger info)
    """
    return ASK_USER_TRIGGERS.get(trigger, ASK_USER_TRIGGERS["unknown"])
