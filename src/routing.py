"""
Routing Factory for LangGraph Verdict-Based Routing

This module provides a factory function for creating routing functions
that route based on verdict fields in the ReproState. This eliminates
the repetitive boilerplate code in graph.py.

Each routing function:
1. Checks for None verdict (logs error, escalates to ask_user)
2. Maps verdict values to route names
3. Checks revision/failure counts against limits
4. Saves checkpoints before escalation

═══════════════════════════════════════════════════════════════════════════════
DESIGN RATIONALE
═══════════════════════════════════════════════════════════════════════════════

The original graph.py had 8 routing functions with nearly identical structure:
- Check verdict is not None
- Log error if None
- Save checkpoint before escalation
- Route based on verdict with optional count limits

This factory extracts that pattern, reducing ~280 lines to ~40 lines of
configuration-based router creation.
"""

import logging
from typing import Callable, Dict, List, Optional, TypedDict, Any, Literal

from schemas.state import ReproState, save_checkpoint


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Routing Decision Logging
# ═══════════════════════════════════════════════════════════════════════

def _log_routing_decision(state: "ReproState", checkpoint_prefix: str, verdict: str, target_route: str) -> None:
    """Log routing decision with relevant context at INFO level.
    
    Provides visibility into the workflow by logging each routing decision
    along with the relevant feedback or issues that influenced it.
    """
    context_parts = []
    
    # Get feedback/issues based on review type
    if checkpoint_prefix == "plan_review":
        issues = state.get("reviewer_issues", [])
        feedback = state.get("planner_feedback") or state.get("reviewer_feedback") or ""
        feedback_str = str(feedback)
        
        if issues and verdict == "needs_revision":
            # Show actual issues if available
            issues_str = "; ".join(str(i)[:60] for i in issues[:2])
            context_parts.append(f"issues: [{issues_str}]")
        elif feedback_str and verdict == "needs_revision":
            # Extract the actual problem from feedback (skip praise)
            issue_excerpt = None
            for marker in ["PLAN_ISSUE:", "Issue:", "Problem:", "Missing:", "Error:", "must ", "should ", "needs "]:
                idx = feedback_str.lower().find(marker.lower())
                if idx >= 0:
                    # Show from the issue marker
                    issue_excerpt = feedback_str[idx:idx+120].replace('\n', ' ')
                    break
            if issue_excerpt:
                context_parts.append(f"{issue_excerpt}...")
            elif len(feedback_str) > 100:
                # No marker found - show end of feedback (usually where issues are)
                context_parts.append(f"...{feedback_str[-100:].replace(chr(10), ' ')}")
            
    elif checkpoint_prefix == "design_review":
        issues = state.get("reviewer_issues", [])
        if issues and verdict == "needs_revision":
            issues_preview = issues[:2] if isinstance(issues, list) else [str(issues)[:80]]
            context_parts.append(f"issues: {issues_preview}")
            
    elif checkpoint_prefix == "code_review":
        issues = state.get("reviewer_issues", [])
        if issues and verdict == "needs_revision":
            issues_preview = issues[:2] if isinstance(issues, list) else [str(issues)[:80]]
            context_parts.append(f"issues: {issues_preview}")
            
    elif checkpoint_prefix == "execution":
        if verdict == "fail":
            error = state.get("execution_error")
            if error:
                error_str = str(error)
                context_parts.append(f"error: {error_str[:80]}{'...' if len(error_str) > 80 else ''}")
        elif verdict == "warning":
            warnings = state.get("execution_warnings", [])
            if warnings:
                context_parts.append(f"{len(warnings)} warning(s)")
            
    elif checkpoint_prefix == "physics":
        issues = state.get("physics_issues", [])
        if issues and verdict in ["fail", "warning", "design_flaw"]:
            issues_preview = issues[:2] if isinstance(issues, list) else [str(issues)[:80]]
            context_parts.append(f"issues: {issues_preview}")
    
    elif checkpoint_prefix == "comparison":
        match_score = state.get("match_score")
        if match_score is not None:
            context_parts.append(f"match_score: {match_score}")
        if verdict == "needs_revision":
            feedback = state.get("comparison_feedback")
            if feedback:
                feedback_str = str(feedback)
                context_parts.append(f"feedback: {feedback_str[:60]}{'...' if len(feedback_str) > 60 else ''}")
    
    # Format the log message with emoji for quick scanning
    emoji = "✅" if verdict in ["approve", "pass"] else "🔄" if verdict == "needs_revision" else "⚠️" if verdict == "warning" else "❌"
    context_str = f" ({', '.join(context_parts)})" if context_parts else ""
    
    logger.info(f"{emoji} {checkpoint_prefix}: verdict={verdict} → {target_route}{context_str}")


# ═══════════════════════════════════════════════════════════════════════
# Route Type Definitions
# ═══════════════════════════════════════════════════════════════════════
#
# These Literal types define all valid route names in the graph.
# Using Literal types enables:
# - Type checker catches route name typos
# - IDE autocompletion for route names
# - Self-documenting code (valid routes are explicit)
# ═══════════════════════════════════════════════════════════════════════

# All possible node/route names in the LangGraph state machine
RouteType = Literal[
    "adapt_prompts",
    "planning",
    "plan_review",
    "select_stage",
    "design",
    "design_review",
    "generate_code",
    "code_review",
    "run_code",
    "execution_check",
    "physics_check",
    "analyze",
    "comparison_check",
    "supervisor",
    "ask_user",
    "generate_report",
    "handle_backtrack",
    "material_checkpoint",
]

# Verdict literals for each review/validation node
PlanReviewVerdict = Literal["approve", "needs_revision"]
DesignReviewVerdict = Literal["approve", "needs_revision"]
CodeReviewVerdict = Literal["approve", "needs_revision"]
ExecutionVerdict = Literal["pass", "warning", "fail"]
PhysicsVerdict = Literal["pass", "warning", "fail", "design_flaw"]
ComparisonVerdict = Literal["approve", "needs_revision"]

# Type aliases for router function signatures
RouterFunction = Callable[[ReproState], RouteType]


# ═══════════════════════════════════════════════════════════════════════
# Type Definitions
# ═══════════════════════════════════════════════════════════════════════

class CountLimitConfig(TypedDict, total=False):
    """Configuration for count-limited routing."""
    count_field: str  # State field containing current count (e.g., "code_revision_count")
    max_count_key: str  # Key in runtime_config for max count (e.g., "max_code_revisions")
    default_max: int  # Default max if not in runtime_config
    route_on_limit: str  # Route when count limit exceeded (default: "ask_user")


class VerdictRouteConfig(TypedDict, total=False):
    """Configuration for a single verdict route."""
    route: str  # Target route name
    count_limit: Optional[CountLimitConfig]  # Optional count limiting


# ═══════════════════════════════════════════════════════════════════════
# Factory Function
# ═══════════════════════════════════════════════════════════════════════

def create_verdict_router(
    verdict_field: str,
    routes: Dict[str, VerdictRouteConfig],
    checkpoint_prefix: str,
    pass_through_verdicts: Optional[List[str]] = None,
) -> RouterFunction:
    """
    Factory function for creating verdict-based routing functions.
    
    Eliminates repetitive boilerplate by generating routing functions
    from configuration.
    
    Args:
        verdict_field: State field containing the verdict (e.g., "execution_verdict")
        routes: Mapping of verdict values to route configurations
        checkpoint_prefix: Prefix for checkpoint names (e.g., "execution")
        pass_through_verdicts: Optional list of verdicts that route directly without count checks
        
    Returns:
        A routing function suitable for use with LangGraph conditional_edges
        
    Example:
        >>> route_after_code_review = create_verdict_router(
        ...     verdict_field="last_code_review_verdict",
        ...     routes={
        ...         "approve": {"route": "run_code"},
        ...         "needs_revision": {
        ...             "route": "generate_code",
        ...             "count_limit": {
        ...                 "count_field": "code_revision_count",
        ...                 "max_count_key": "max_code_revisions",
        ...                 "default_max": 3,
        ...             }
        ...         },
        ...     },
        ...     checkpoint_prefix="code_review",
        ... )
    """
    pass_through = set(pass_through_verdicts or [])
    
    def router(state: ReproState) -> RouteType:
        verdict = state.get(verdict_field)
        
        # ═══════════════════════════════════════════════════════════════════════
        # HANDLE NONE VERDICT: Log error and escalate to user
        # ═══════════════════════════════════════════════════════════════════════
        if verdict is None:
            logger.error(
                f"{verdict_field} is None - node may not have run or failed. "
                f"Escalating to user for guidance."
            )
            save_checkpoint(state, f"before_ask_user_{checkpoint_prefix}_error")
            return "ask_user"
        
        # ═══════════════════════════════════════════════════════════════════════
        # HANDLE INVALID VERDICT TYPES: Only strings are valid verdicts
        # ═══════════════════════════════════════════════════════════════════════
        if not isinstance(verdict, str):
            logger.error(
                f"{verdict_field} has invalid type {type(verdict).__name__}, expected str. "
                f"Value: {verdict}. Escalating to user for guidance."
            )
            save_checkpoint(state, f"before_ask_user_{checkpoint_prefix}_error")
            return "ask_user"
        
        # ═══════════════════════════════════════════════════════════════════════
        # ROUTE BASED ON VERDICT
        # ═══════════════════════════════════════════════════════════════════════
        if verdict in routes:
            route_config = routes[verdict]
            target_route = route_config.get("route", "ask_user")
            
            # Check if this verdict has count limiting
            count_limit = route_config.get("count_limit")
            if count_limit and verdict not in pass_through:
                # Get limit configuration
                count_field = count_limit.get("count_field", "")
                max_count_key = count_limit.get("max_count_key", "")
                default_max = count_limit.get("default_max", 3)
                route_on_limit = count_limit.get("route_on_limit", "ask_user")
                
                # Check current count against limit
                runtime_config = state.get("runtime_config") or {}
                max_count = runtime_config.get(max_count_key, default_max)
                raw_count = state.get(count_field)
                # Handle invalid count types (None, strings, etc.) by defaulting to 0
                try:
                    current_count = int(raw_count) if raw_count is not None else 0
                except (TypeError, ValueError):
                    logger.warning(
                        f"{checkpoint_prefix}: {count_field} has invalid value {raw_count!r}, treating as 0"
                    )
                    current_count = 0
                
                if current_count >= max_count:
                    logger.warning(
                        f"{checkpoint_prefix}: {count_field}={current_count} >= {max_count_key}={max_count}, "
                        f"escalating to {route_on_limit}"
                    )
                    save_checkpoint(state, f"before_ask_user_{checkpoint_prefix}_limit")
                    return route_on_limit
            
            # Log the routing decision with context
            _log_routing_decision(state, checkpoint_prefix, verdict, target_route)
            
            return target_route
        
        # ═══════════════════════════════════════════════════════════════════════
        # FALLBACK: Unknown verdict, escalate to user
        # ═══════════════════════════════════════════════════════════════════════
        logger.warning(
            f"{verdict_field}='{verdict}' is not a recognized verdict. "
            f"Escalating to ask_user."
        )
        save_checkpoint(state, f"before_ask_user_{checkpoint_prefix}_fallback")
        return "ask_user"
    
    return router


# ═══════════════════════════════════════════════════════════════════════
# Pre-configured Routers for graph.py
# ═══════════════════════════════════════════════════════════════════════
#
# These are imported by graph.py to replace the verbose routing functions.
# Each router is configured to match the original behavior exactly.
# ═══════════════════════════════════════════════════════════════════════

from schemas.state import (
    MAX_DESIGN_REVISIONS,
    MAX_CODE_REVISIONS,
    MAX_EXECUTION_FAILURES,
    MAX_PHYSICS_FAILURES,
    MAX_ANALYSIS_REVISIONS,
    MAX_REPLANS,
)


# ───────────────────────────────────────────────────────────────────────
# Plan Review Router
# Routes: select_stage (approve) | planning (revision) | ask_user (limit/error)
# ───────────────────────────────────────────────────────────────────────
route_after_plan_review: Callable[[ReproState], Literal["select_stage", "planning", "ask_user"]]
route_after_plan_review = create_verdict_router(
    verdict_field="last_plan_review_verdict",
    routes={
        "approve": {"route": "select_stage"},
        "needs_revision": {
            "route": "planning",
            "count_limit": {
                "count_field": "replan_count",
                "max_count_key": "max_replans",
                "default_max": MAX_REPLANS,
            }
        },
    },
    checkpoint_prefix="plan_review",
)


# ───────────────────────────────────────────────────────────────────────
# Design Review Router
# Routes: generate_code (approve) | design (revision) | ask_user (limit/error)
# ───────────────────────────────────────────────────────────────────────
route_after_design_review: Callable[[ReproState], Literal["generate_code", "design", "ask_user"]]
route_after_design_review = create_verdict_router(
    verdict_field="last_design_review_verdict",
    routes={
        "approve": {"route": "generate_code"},
        "needs_revision": {
            "route": "design",
            "count_limit": {
                "count_field": "design_revision_count",
                "max_count_key": "max_design_revisions",
                "default_max": MAX_DESIGN_REVISIONS,
            }
        },
    },
    checkpoint_prefix="design_review",
)


# ───────────────────────────────────────────────────────────────────────
# Code Review Router
# Routes: run_code (approve) | generate_code (revision) | ask_user (limit/error)
# ───────────────────────────────────────────────────────────────────────
route_after_code_review: Callable[[ReproState], Literal["run_code", "generate_code", "ask_user"]]
route_after_code_review = create_verdict_router(
    verdict_field="last_code_review_verdict",
    routes={
        "approve": {"route": "run_code"},
        "needs_revision": {
            "route": "generate_code",
            "count_limit": {
                "count_field": "code_revision_count",
                "max_count_key": "max_code_revisions",
                "default_max": MAX_CODE_REVISIONS,
            }
        },
    },
    checkpoint_prefix="code_review",
)


# ───────────────────────────────────────────────────────────────────────
# Execution Check Router
# Routes: physics_check (pass/warning) | generate_code (fail) | ask_user (limit/error)
# ───────────────────────────────────────────────────────────────────────
route_after_execution_check: Callable[[ReproState], Literal["physics_check", "generate_code", "ask_user"]]
route_after_execution_check = create_verdict_router(
    verdict_field="execution_verdict",
    routes={
        "pass": {"route": "physics_check"},
        "warning": {"route": "physics_check"},
        "fail": {
            "route": "generate_code",
            "count_limit": {
                "count_field": "execution_failure_count",
                "max_count_key": "max_execution_failures",
                "default_max": MAX_EXECUTION_FAILURES,
            }
        },
    },
    checkpoint_prefix="execution",
    pass_through_verdicts=["pass", "warning"],
)


# ───────────────────────────────────────────────────────────────────────
# Physics Check Router
# Routes: analyze (pass/warning) | generate_code (fail) | design (design_flaw) | ask_user (limit/error)
# ───────────────────────────────────────────────────────────────────────
route_after_physics_check: Callable[[ReproState], Literal["analyze", "generate_code", "design", "ask_user"]]
route_after_physics_check = create_verdict_router(
    verdict_field="physics_verdict",
    routes={
        "pass": {"route": "analyze"},
        "warning": {"route": "analyze"},
        "fail": {
            "route": "generate_code",
            "count_limit": {
                "count_field": "physics_failure_count",
                "max_count_key": "max_physics_failures",
                "default_max": MAX_PHYSICS_FAILURES,
            }
        },
        "design_flaw": {
            "route": "design",
            "count_limit": {
                "count_field": "design_revision_count",
                "max_count_key": "max_design_revisions",
                "default_max": MAX_DESIGN_REVISIONS,
            }
        },
    },
    checkpoint_prefix="physics",
    pass_through_verdicts=["pass", "warning"],
)


# ───────────────────────────────────────────────────────────────────────
# Comparison Check Router
# Routes: supervisor (approve/limit) | analyze (revision)
# Note: This router routes to supervisor (not ask_user) when limit is reached
# ───────────────────────────────────────────────────────────────────────
route_after_comparison_check: Callable[[ReproState], Literal["supervisor", "analyze", "ask_user"]]
route_after_comparison_check = create_verdict_router(
    verdict_field="comparison_verdict",
    routes={
        "approve": {"route": "supervisor"},
        "needs_revision": {
            "route": "analyze",
            "count_limit": {
                "count_field": "analysis_revision_count",
                "max_count_key": "max_analysis_revisions",
                "default_max": MAX_ANALYSIS_REVISIONS,
                "route_on_limit": "ask_user",  # Escalate to user like other limits
            }
        },
    },
    checkpoint_prefix="comparison",
)

