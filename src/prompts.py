"""
Prompt Builder - Constructs agent prompts with runtime context injection.

This module handles loading prompt templates and injecting:
1. Constants from state.py (thresholds, limits, etc.)
2. Dynamic state (paper text, figures, current stage, etc.)
3. Cross-agent context (feedback, assumptions, etc.)

The goal is to maintain a single source of truth for constants in code
while making them available to LLM prompts at runtime.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# Import canonical constants from state
from schemas.state import (
    DISCREPANCY_THRESHOLDS,
    format_thresholds_table,
    MAX_DESIGN_REVISIONS,
    MAX_ANALYSIS_REVISIONS,
    MAX_REPLANS,
    DEFAULT_STAGE_BUDGETS,
)


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

# Path to prompts directory (relative to this file)
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Agents and their prompt files
AGENT_PROMPTS = {
    "prompt_adaptor": "prompt_adaptor_agent.md",
    "planner": "planner_agent.md",
    "simulation_designer": "simulation_designer_agent.md",
    "code_generator": "code_generator_agent.md",
    "code_reviewer": "code_reviewer_agent.md",
    "execution_validator": "execution_validator_agent.md",
    "physics_sanity": "physics_sanity_agent.md",
    "results_analyzer": "results_analyzer_agent.md",
    "comparison_validator": "comparison_validator_agent.md",
    "supervisor": "supervisor_agent.md",
}

# Global rules file (prepended to all agent prompts)
GLOBAL_RULES_FILE = "global_rules.md"


# ═══════════════════════════════════════════════════════════════════════
# Prompt Template Loading
# ═══════════════════════════════════════════════════════════════════════

def load_prompt_template(filename: str) -> str:
    """
    Load a raw prompt template from the prompts directory.
    
    Args:
        filename: Name of the prompt file (e.g., "planner_agent.md")
        
    Returns:
        Raw template string
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    
    return path.read_text(encoding='utf-8')


def load_global_rules() -> str:
    """Load the global rules that apply to all agents."""
    return load_prompt_template(GLOBAL_RULES_FILE)


def load_agent_prompt(agent_name: str) -> str:
    """
    Load an agent's prompt template by agent name.
    
    Args:
        agent_name: Short name (e.g., "planner", "code_generator")
        
    Returns:
        Raw template string
    """
    if agent_name not in AGENT_PROMPTS:
        raise ValueError(f"Unknown agent: {agent_name}. Available: {list(AGENT_PROMPTS.keys())}")
    
    return load_prompt_template(AGENT_PROMPTS[agent_name])


# ═══════════════════════════════════════════════════════════════════════
# Placeholder Injection
# ═══════════════════════════════════════════════════════════════════════

def inject_constants(template: str) -> str:
    """
    Inject constant values from state.py into a prompt template.
    
    Replaces placeholders like {THRESHOLDS_TABLE} with actual values
    generated from the canonical source in state.py.
    
    Args:
        template: Prompt template with placeholders
        
    Returns:
        Template with placeholders replaced
    """
    # Define all constant placeholders and their values
    constants = {
        "{THRESHOLDS_TABLE}": format_thresholds_table(),
        "{MAX_DESIGN_REVISIONS}": str(MAX_DESIGN_REVISIONS),
        "{MAX_ANALYSIS_REVISIONS}": str(MAX_ANALYSIS_REVISIONS),
        "{MAX_REPLANS}": str(MAX_REPLANS),
    }
    
    # Replace each placeholder
    result = template
    for placeholder, value in constants.items():
        result = result.replace(placeholder, value)
    
    return result


def inject_state_context(
    template: str,
    agent_name: str,
    state: Dict[str, Any]
) -> str:
    """
    Inject state-dependent context into a prompt template.
    
    Different agents receive different context based on their needs:
    - PlannerAgent: Full paper text, figures
    - SimulationDesigner: Current stage, assumptions
    - ResultsAnalyzer: Outputs, target figures
    - etc.
    
    Args:
        template: Prompt template (after constant injection)
        agent_name: Which agent this prompt is for
        state: Current ReproState
        
    Returns:
        Template with state context appended
    """
    context_section = ""
    
    if agent_name == "planner":
        context_section = _build_planner_context(state)
    elif agent_name == "simulation_designer":
        context_section = _build_designer_context(state)
    elif agent_name == "code_generator":
        context_section = _build_generator_context(state)
    elif agent_name == "results_analyzer":
        context_section = _build_analyzer_context(state)
    elif agent_name == "supervisor":
        context_section = _build_supervisor_context(state)
    elif agent_name == "prompt_adaptor":
        context_section = _build_adaptor_context(state)
    # Other agents get minimal context (most info is in their prompts)
    
    if context_section:
        return template + "\n\n" + context_section
    return template


# ═══════════════════════════════════════════════════════════════════════
# Agent-Specific Context Builders
# ═══════════════════════════════════════════════════════════════════════

def _build_planner_context(state: Dict[str, Any]) -> str:
    """Build context section for PlannerAgent."""
    figures_list = _format_figures(state.get("paper_figures", []))
    
    return f"""
═══════════════════════════════════════════════════════════════════════
CURRENT PAPER (injected at runtime)
═══════════════════════════════════════════════════════════════════════

**Paper ID**: {state.get('paper_id', 'unknown')}
**Title**: {state.get('paper_title', 'Unknown')}
**Domain**: {state.get('paper_domain', 'other')}

### Paper Text:

{state.get('paper_text', '[No paper text provided]')}

### Figures to Reproduce ({len(state.get('paper_figures', []))} total):

{figures_list}
"""


def _build_designer_context(state: Dict[str, Any]) -> str:
    """Build context section for SimulationDesignerAgent."""
    stage_info = _get_current_stage_info(state)
    assumptions = _format_assumptions(state)
    
    return f"""
═══════════════════════════════════════════════════════════════════════
CURRENT TASK (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Stage Information:
- **Stage ID**: {state.get('current_stage_id', 'unknown')}
- **Stage Type**: {state.get('current_stage_type', 'unknown')}
- **Design Revision**: {state.get('design_revision_count', 0) + 1} of {MAX_DESIGN_REVISIONS}

### Stage Requirements:
{stage_info}

### Current Assumptions:
{assumptions}

### Previous Feedback (if revising):
{state.get('reviewer_feedback', 'N/A - first design attempt')}
"""


def _build_generator_context(state: Dict[str, Any]) -> str:
    """Build context section for CodeGeneratorAgent."""
    return f"""
═══════════════════════════════════════════════════════════════════════
CURRENT TASK (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Design to Implement:

{state.get('design_description', '[No design provided]')}

### Previous Feedback (if revising):
{state.get('reviewer_feedback', 'N/A - first code attempt')}

### Performance Estimate:
{json.dumps(state.get('performance_estimate', {}), indent=2) if state.get('performance_estimate') else 'N/A'}
"""


def _build_analyzer_context(state: Dict[str, Any]) -> str:
    """Build context section for ResultsAnalyzerAgent."""
    outputs = state.get("stage_outputs", {})
    target_figures = _get_target_figures_for_stage(state)
    
    return f"""
═══════════════════════════════════════════════════════════════════════
ANALYSIS TASK (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Simulation Outputs:

**Exit Code**: {outputs.get('exit_code', 'unknown')}
**Runtime**: {outputs.get('runtime_seconds', 0):.1f} seconds

**Output Files**:
{chr(10).join('- ' + f for f in outputs.get('files', [])) or 'None'}

**Stdout** (last 50 lines):
```
{_truncate_output(outputs.get('stdout', ''), 50)}
```

**Stderr**:
```
{outputs.get('stderr', '')}
```

### Target Figures to Compare:

{target_figures}

### Digitized Data Available:
{_list_digitized_data(state)}
"""


def _build_supervisor_context(state: Dict[str, Any]) -> str:
    """Build context section for SupervisorAgent."""
    progress_summary = _format_progress_summary(state)
    
    return f"""
═══════════════════════════════════════════════════════════════════════
CURRENT STATE (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Overall Progress:
{progress_summary}

### Validation Hierarchy Status:
{json.dumps(state.get('validation_hierarchy', {}), indent=2)}

### Recent Figure Comparisons:
{_format_recent_comparisons(state)}

### Runtime Budget:
- **Remaining**: {state.get('runtime_budget_remaining_seconds', 0) / 60:.1f} minutes
- **Total Used**: {state.get('total_runtime_seconds', 0) / 60:.1f} minutes

### Pending User Questions:
{chr(10).join('- ' + q for q in state.get('pending_user_questions', [])) or 'None'}
"""


def _build_adaptor_context(state: Dict[str, Any]) -> str:
    """Build context section for PromptAdaptorAgent."""
    # Adaptor gets a quick summary of the paper for domain detection
    paper_text = state.get('paper_text', '')
    # Truncate for initial analysis
    summary_length = min(5000, len(paper_text))
    
    return f"""
═══════════════════════════════════════════════════════════════════════
PAPER SUMMARY (for prompt adaptation analysis)
═══════════════════════════════════════════════════════════════════════

**Paper ID**: {state.get('paper_id', 'unknown')}
**Title**: {state.get('paper_title', 'Unknown')}
**Declared Domain**: {state.get('paper_domain', 'other')}

### Paper Text (first {summary_length} chars for analysis):

{paper_text[:summary_length]}

{'[... truncated for initial analysis ...]' if len(paper_text) > summary_length else ''}

### Available Agents to Adapt:
{chr(10).join('- ' + name for name in AGENT_PROMPTS.keys())}
"""


# ═══════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════

def _format_figures(figures: List[Dict[str, Any]]) -> str:
    """Format figure list for prompt display."""
    if not figures:
        return "No figures provided"
    
    lines = []
    for fig in figures:
        fig_id = fig.get('id', 'unknown')
        desc = fig.get('description', 'No description')
        has_digitized = "✓" if fig.get('digitized_data_path') else "✗"
        lines.append(f"- **{fig_id}** [digitized: {has_digitized}]: {desc}")
    
    return "\n".join(lines)


def _format_assumptions(state: Dict[str, Any]) -> str:
    """Format current assumptions for display."""
    assumptions = state.get("assumptions", {})
    if not assumptions:
        return "No assumptions defined yet"
    
    # Format global assumptions
    global_assumptions = assumptions.get("global_assumptions", {})
    lines = ["**Global Assumptions:**"]
    
    for category, items in global_assumptions.items():
        if items:
            lines.append(f"\n*{category.replace('_', ' ').title()}*:")
            for item in items:
                if isinstance(item, dict):
                    lines.append(f"  - {item.get('description', str(item))}")
                else:
                    lines.append(f"  - {item}")
    
    return "\n".join(lines) if len(lines) > 1 else "No assumptions defined yet"


def _get_current_stage_info(state: Dict[str, Any]) -> str:
    """Get info about the current stage from the plan."""
    plan = state.get("plan", {})
    stages = plan.get("stages", [])
    current_id = state.get("current_stage_id")
    
    for stage in stages:
        if stage.get("stage_id") == current_id:
            return json.dumps(stage, indent=2)
    
    return "Stage not found in plan"


def _get_target_figures_for_stage(state: Dict[str, Any]) -> str:
    """Get target figures for the current stage."""
    plan = state.get("plan", {})
    stages = plan.get("stages", [])
    current_id = state.get("current_stage_id")
    
    for stage in stages:
        if stage.get("stage_id") == current_id:
            targets = stage.get("target_figures", [])
            if targets:
                # Also get figure details
                figures = state.get("paper_figures", [])
                lines = []
                for target in targets:
                    fig = next((f for f in figures if f.get('id') == target), None)
                    if fig:
                        lines.append(f"- **{target}**: {fig.get('description', 'No description')}")
                        if fig.get('image_path'):
                            lines.append(f"  Image: {fig['image_path']}")
                        if fig.get('digitized_data_path'):
                            lines.append(f"  Digitized: {fig['digitized_data_path']}")
                    else:
                        lines.append(f"- **{target}**: [figure details not found]")
                return "\n".join(lines)
            return "No target figures for this stage"
    
    return "Stage not found"


def _list_digitized_data(state: Dict[str, Any]) -> str:
    """List any available digitized data files."""
    figures = state.get("paper_figures", [])
    digitized = [f for f in figures if f.get('digitized_data_path')]
    
    if not digitized:
        return "No digitized data available"
    
    lines = []
    for fig in digitized:
        lines.append(f"- {fig['id']}: {fig['digitized_data_path']}")
    
    return "\n".join(lines)


def _format_progress_summary(state: Dict[str, Any]) -> str:
    """Format overall progress for supervisor."""
    progress = state.get("progress", {})
    stage_progress = progress.get("stage_progress", {})
    
    if not stage_progress:
        return "No stages started yet"
    
    lines = []
    for stage_id, info in stage_progress.items():
        status = info.get("status", "unknown")
        revision = info.get("revision_count", 0)
        lines.append(f"- {stage_id}: {status} (revisions: {revision})")
    
    return "\n".join(lines)


def _format_recent_comparisons(state: Dict[str, Any]) -> str:
    """Format recent figure comparisons."""
    comparisons = state.get("figure_comparisons", [])
    
    if not comparisons:
        return "No comparisons yet"
    
    # Show last 3 comparisons
    recent = comparisons[-3:]
    lines = []
    for comp in recent:
        fig_id = comp.get("figure_id", "unknown")
        classification = comp.get("classification", "unknown")
        confidence = comp.get("confidence", 0)
        lines.append(f"- {fig_id}: {classification} (confidence: {confidence:.0%})")
    
    return "\n".join(lines)


def _truncate_output(text: str, max_lines: int) -> str:
    """Truncate text to last N lines."""
    if not text:
        return "[no output]"
    
    lines = text.strip().split('\n')
    if len(lines) <= max_lines:
        return text
    
    return f"[... {len(lines) - max_lines} lines omitted ...]\n" + '\n'.join(lines[-max_lines:])


# ═══════════════════════════════════════════════════════════════════════
# Main Prompt Building Function
# ═══════════════════════════════════════════════════════════════════════

def build_prompt(
    agent_name: str,
    state: Dict[str, Any],
    include_global_rules: bool = True
) -> str:
    """
    Build a complete prompt for an agent with all injections.
    
    This is the main entry point for prompt construction. It:
    1. Loads the global rules (if requested)
    2. Loads the agent-specific prompt template
    3. Injects constants from state.py
    4. Injects state-dependent context
    
    Args:
        agent_name: Short name of the agent (e.g., "planner")
        state: Current ReproState dictionary
        include_global_rules: Whether to prepend global rules
        
    Returns:
        Complete prompt string ready for LLM
        
    Example:
        prompt = build_prompt("planner", state)
        response = llm.invoke(prompt)
    """
    parts = []
    
    # 1. Global rules (always first)
    if include_global_rules:
        global_rules = load_global_rules()
        global_rules = inject_constants(global_rules)
        parts.append(global_rules)
        parts.append("\n\n" + "="*75 + "\n\n")
    
    # 2. Agent-specific prompt
    agent_prompt = load_agent_prompt(agent_name)
    agent_prompt = inject_constants(agent_prompt)
    agent_prompt = inject_state_context(agent_prompt, agent_name, state)
    parts.append(agent_prompt)
    
    return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Utility Functions for Testing
# ═══════════════════════════════════════════════════════════════════════

def preview_injected_thresholds() -> str:
    """Preview what the thresholds table looks like after injection."""
    return format_thresholds_table()


def list_placeholders_in_prompts() -> Dict[str, List[str]]:
    """
    Scan all prompts for placeholders that need injection.
    
    Returns dict mapping prompt files to list of placeholders found.
    """
    import re
    
    placeholder_pattern = re.compile(r'\{[A-Z_]+\}')
    results = {}
    
    # Check all prompt files
    for name, filename in AGENT_PROMPTS.items():
        try:
            content = load_prompt_template(filename)
            placeholders = placeholder_pattern.findall(content)
            if placeholders:
                results[filename] = placeholders
        except FileNotFoundError:
            results[filename] = ["[FILE NOT FOUND]"]
    
    # Check global rules
    try:
        content = load_global_rules()
        placeholders = placeholder_pattern.findall(content)
        if placeholders:
            results[GLOBAL_RULES_FILE] = placeholders
    except FileNotFoundError:
        results[GLOBAL_RULES_FILE] = ["[FILE NOT FOUND]"]
    
    return results


if __name__ == "__main__":
    # Test the prompt building
    print("=== Prompt Builder Test ===\n")
    
    # Preview thresholds
    print("Thresholds Table (from state.py):")
    print(preview_injected_thresholds())
    print()
    
    # List all placeholders
    print("Placeholders found in prompts:")
    for filename, placeholders in list_placeholders_in_prompts().items():
        print(f"  {filename}: {placeholders}")
    print()
    
    # Test building a prompt with minimal state
    test_state = {
        "paper_id": "test_paper",
        "paper_title": "Test Paper Title",
        "paper_domain": "plasmonics",
        "paper_text": "This is a test paper about plasmonics...",
        "paper_figures": [
            {"id": "Fig1", "description": "Test figure", "image_path": "fig1.png"}
        ]
    }
    
    print("Building planner prompt (first 500 chars):")
    try:
        prompt = build_prompt("planner", test_state)
        print(prompt[:500])
        print("...\n")
        print(f"Total prompt length: {len(prompt)} characters")
    except Exception as e:
        print(f"Error: {e}")

