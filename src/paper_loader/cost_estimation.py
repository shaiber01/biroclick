"""
Token and cost estimation for paper reproduction.

This module provides functions to estimate:
- Token counts for paper text and figures
- API costs for running the reproduction workflow
- Paper length warnings
"""

from typing import Dict, Any, List

from .config import (
    CHARS_PER_TOKEN,
    TOKENS_PER_FIGURE,
    INPUT_COST_PER_MILLION,
    OUTPUT_COST_PER_MILLION,
    PAPER_LENGTH_NORMAL,
    PAPER_LENGTH_LONG,
    PAPER_LENGTH_VERY_LONG,
)


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate (~4 chars per token for English text).
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    return len(text) // CHARS_PER_TOKEN


def check_paper_length(text: str, label: str = "Paper") -> List[str]:
    """
    Check paper text length and return warnings if too long.
    
    Args:
        text: The paper text to check
        label: Label for the text (e.g., "Paper", "Supplementary")
        
    Returns:
        List of warning strings (empty if length is normal)
    """
    warnings: List[str] = []
    char_count = len(text)
    token_estimate = estimate_tokens(text)
    
    if char_count > PAPER_LENGTH_VERY_LONG:
        warnings.append(
            f"{label} is VERY LONG: {char_count:,} chars (~{token_estimate:,} tokens). "
            f"This may exceed LLM context limits and significantly increase costs. "
            f"Consider removing references, acknowledgments, or non-essential sections."
        )
    elif char_count > PAPER_LENGTH_LONG:
        warnings.append(
            f"{label} is long: {char_count:,} chars (~{token_estimate:,} tokens). "
            f"Consider trimming references section to reduce costs."
        )
    
    return warnings


def estimate_token_cost(paper_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimate token usage and cost for reproducing a paper.
    
    This provides a rough estimate based on:
    - Paper text length
    - Number of figures (vision models charge per image)
    - Expected number of agent calls
    
    NOTE: This is a ROUGH estimate. Actual costs depend on:
    - Number of revisions needed
    - Model used (Claude, GPT-4, etc.) and current pricing
    - Complexity of the reproduction
    
    Args:
        paper_input: Paper input dictionary
        
    Returns:
        Dictionary with token and cost estimates
    """
    # Text tokens (rough estimate: ~4 chars per token)
    text_tokens = len(paper_input.get("paper_text", "")) / CHARS_PER_TOKEN
    
    # Supplementary text if present
    supplementary = paper_input.get("supplementary", {})
    if supplementary.get("supplementary_text"):
        text_tokens += len(supplementary["supplementary_text"]) / CHARS_PER_TOKEN
    
    # Image tokens (approximate, varies by model)
    figures = paper_input.get("figures", [])
    supp_figures = supplementary.get("supplementary_figures", [])
    total_figures = len(figures) + len(supp_figures)
    image_tokens = total_figures * TOKENS_PER_FIGURE
    
    # Estimate agent calls by workflow phase
    # Planning phase: 2 calls with full paper text
    planner_input_tokens = 2 * text_tokens
    
    # Per stage estimates (assume 4 stages average)
    num_stages = max(4, len(figures))  # At least one stage per key figure
    
    # Each stage involves:
    # - Design: 1 call with ~30% of paper context
    # - CodeGen: 2 calls (initial + 1 revision average)
    # - Review: 2 calls
    # - Analysis: 1 call with images
    stage_text_fraction = 0.3  # Only relevant parts of paper needed
    per_stage_text = text_tokens * stage_text_fraction
    per_stage_input = (
        per_stage_text +  # Design
        per_stage_text * 2 +  # CodeGen (2 calls)
        per_stage_text * 2 +  # Review (2 calls)
        per_stage_text + image_tokens / num_stages  # Analysis with figure
    )
    
    total_stages_input = per_stage_input * num_stages
    
    # Supervisor: 1 call per stage decision
    supervisor_input = num_stages * (per_stage_text * 0.5)
    
    # Report generation: 1 call with summaries
    report_input = text_tokens * 0.2 + image_tokens
    
    # Total input tokens
    total_input_estimate = (
        planner_input_tokens +
        total_stages_input +
        supervisor_input +
        report_input
    )
    
    # Output tokens (typically 20-30% of input for this use case)
    total_output_estimate = total_input_estimate * 0.25
    
    # Cost calculation
    input_cost = total_input_estimate * INPUT_COST_PER_MILLION / 1_000_000
    output_cost = total_output_estimate * OUTPUT_COST_PER_MILLION / 1_000_000
    total_cost = input_cost + output_cost
    
    return {
        "estimated_input_tokens": int(total_input_estimate),
        "estimated_output_tokens": int(total_output_estimate),
        "estimated_total_tokens": int(total_input_estimate + total_output_estimate),
        "estimated_cost_usd": round(total_cost, 2),
        "cost_breakdown": {
            "input_cost_usd": round(input_cost, 2),
            "output_cost_usd": round(output_cost, 2),
        },
        "assumptions": {
            "num_figures": total_figures,
            "num_stages_estimated": num_stages,
            "text_chars": len(paper_input.get("paper_text", "")),
            "model_pricing": "approximate LLM rates (verify current pricing)",
        },
        "warning": (
            "This is a rough estimate. Actual costs depend on number of revisions, "
            "model selection, and reproduction complexity. "
            "Budget 2-3x this estimate for complex papers."
        )
    }



