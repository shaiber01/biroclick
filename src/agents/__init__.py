"""
Agents Package - Modular agent node implementations for LangGraph workflow.

This package contains all agent node implementations, split into logical modules:
- planning: plan_node, plan_reviewer_node, adapt_prompts_node
- stage_selection: select_stage_node
- design: simulation_designer_node, design_reviewer_node
- code: code_generator_node, code_reviewer_node
- execution: execution_validator_node, physics_sanity_node
- analysis: results_analyzer_node, comparison_validator_node
- supervision: supervisor_node
- user_interaction: ask_user_node, material_checkpoint_node
- reporting: generate_report_node, handle_backtrack_node

All public node functions are re-exported here for backward compatibility.
Import via: `from src.agents import plan_node, supervisor_node, ...`
"""

# Re-export all node functions from submodules
from .planning import plan_node, plan_reviewer_node, adapt_prompts_node
from .stage_selection import select_stage_node
from .design import simulation_designer_node, design_reviewer_node
from .code import code_generator_node, code_reviewer_node
from .execution import execution_validator_node, physics_sanity_node
from .analysis import results_analyzer_node, comparison_validator_node
from .supervision import supervisor_node
from .user_interaction import ask_user_node, material_checkpoint_node
from .reporting import generate_report_node, handle_backtrack_node

# Re-export helper utilities that may be used externally
from .helpers.metrics import log_agent_call

__all__ = [
    # Planning
    "plan_node",
    "plan_reviewer_node",
    "adapt_prompts_node",
    # Stage selection
    "select_stage_node",
    # Design
    "simulation_designer_node",
    "design_reviewer_node",
    # Code
    "code_generator_node",
    "code_reviewer_node",
    # Execution
    "execution_validator_node",
    "physics_sanity_node",
    # Analysis
    "results_analyzer_node",
    "comparison_validator_node",
    # Supervision
    "supervisor_node",
    # User interaction
    "ask_user_node",
    "material_checkpoint_node",
    # Reporting
    "generate_report_node",
    "handle_backtrack_node",
    # Helpers
    "log_agent_call",
]

