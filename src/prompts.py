"""
Prompt Builder - Constructs agent prompts with runtime context injection.

This module handles loading prompt templates and injecting:
1. Constants from state.py (thresholds, limits, etc.)
2. Dynamic state (paper text, figures, current stage, etc.)
3. Cross-agent context (feedback, assumptions, etc.)

Also provides utilities for:
- Image encoding for Claude's vision API
- Comparison image generation for reports

The goal is to maintain a single source of truth for constants in code
while making them available to LLM prompts at runtime.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import base64
import logging

logger = logging.getLogger(__name__)

# Import canonical constants from state
from schemas.state import (
    DISCREPANCY_THRESHOLDS,
    format_thresholds_table,
    get_validation_hierarchy,
    MAX_DESIGN_REVISIONS,
    MAX_CODE_REVISIONS,
    MAX_ANALYSIS_REVISIONS,
    MAX_REPLANS,
    DEFAULT_STAGE_BUDGETS,
)


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

# Path to prompts directory (relative to this file)
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Path to schemas directory (relative to this file)
SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"

# Agents and their prompt files
AGENT_PROMPTS = {
    "prompt_adaptor": "prompt_adaptor_agent.md",
    "planner": "planner_agent.md",
    "plan_reviewer": "plan_reviewer_agent.md",
    "simulation_designer": "simulation_designer_agent.md",
    "design_reviewer": "design_reviewer_agent.md",
    "code_generator": "code_generator_agent.md",
    "code_reviewer": "code_reviewer_agent.md",
    "execution_validator": "execution_validator_agent.md",
    "physics_sanity": "physics_sanity_agent.md",
    "results_analyzer": "results_analyzer_agent.md",
    "comparison_validator": "comparison_validator_agent.md",
    "supervisor": "supervisor_agent.md",
}

# Agent output schemas (for function calling)
AGENT_OUTPUT_SCHEMAS = {
    "plan_reviewer": "plan_reviewer_output_schema.json",
    "design_reviewer": "design_reviewer_output_schema.json",
    "code_reviewer": "code_reviewer_output_schema.json",
    "execution_validator": "execution_validator_output_schema.json",
    "physics_sanity": "physics_sanity_output_schema.json",
    "results_analyzer": "results_analyzer_output_schema.json",
    "comparison_validator": "comparison_validator_output_schema.json",
    "supervisor": "supervisor_output_schema.json",
}

# Global rules file (prepended to all agent prompts)
GLOBAL_RULES_FILE = "global_rules.md"


# ═══════════════════════════════════════════════════════════════════════
# Image Encoding for Vision API
# ═══════════════════════════════════════════════════════════════════════

def encode_image_for_claude(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Encode an image file for Claude's vision API.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dict with type, source (base64) for Claude message content,
        or None if image cannot be loaded.
        
    Example:
        image_content = encode_image_for_claude("papers/fig3a.png")
        if image_content:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these figures..."},
                    image_content,
                ]
            }]
    """
    path = Path(image_path)
    if not path.exists():
        logger.warning(f"Image not found: {image_path}")
        return None
    
    # Determine media type from extension
    suffix = path.suffix.lower()
    media_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    media_type = media_types.get(suffix)
    if not media_type:
        logger.warning(f"Unsupported image format: {suffix}")
        return None
    
    try:
        with open(path, 'rb') as f:
            data = base64.standard_b64encode(f.read()).decode('utf-8')
        
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            }
        }
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None


def create_comparison_image(
    paper_image_path: str,
    reproduction_image_path: str,
    output_path: str,
    title: str = "Comparison"
) -> bool:
    """
    Create a side-by-side comparison image for reports.
    
    Args:
        paper_image_path: Path to the paper's figure image
        reproduction_image_path: Path to the reproduction image
        output_path: Where to save the comparison image
        title: Title for the comparison (used in figure suptitle)
        
    Returns:
        True if successful, False otherwise.
        
    Example:
        success = create_comparison_image(
            "papers/fig3a.png",
            "outputs/paper_id/stage1_spectrum.png",
            "outputs/paper_id/stage1_comparison.png",
            "Stage 1 - Transmission Spectrum"
        )
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except ImportError:
        logger.error("matplotlib required for comparison images")
        return False
    
    try:
        paper_img = mpimg.imread(paper_image_path)
        repro_img = mpimg.imread(reproduction_image_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].imshow(paper_img)
        axes[0].set_title("Paper Figure", fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(repro_img)
        axes[1].set_title("Reproduction", fontsize=12)
        axes[1].axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Created comparison image: {output_path}")
        return True
        
    except FileNotFoundError as e:
        logger.warning(f"Image file not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to create comparison image: {e}")
        return False


def prepare_vision_comparison_content(
    paper_figure_path: str,
    reproduction_path: str,
    figure_id: str,
    comparison_question: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Prepare content list for a vision-based figure comparison.
    
    Encodes both images and creates appropriate text prompt.
    Returns content list for Claude message and success flag.
    
    Args:
        paper_figure_path: Path to the original paper figure
        reproduction_path: Path to the simulation output figure
        figure_id: Figure ID for reference (e.g., "Fig3a")
        comparison_question: Custom comparison question (optional)
        
    Returns:
        Tuple of (content_list, success_flag)
        - content_list: List of content items for Claude message
        - success_flag: True if both images loaded, False if fallback to text
        
    Example:
        content, vision_ok = prepare_vision_comparison_content(
            "papers/fig3a.png",
            "outputs/stage1_spectrum.png",
            "Fig3a"
        )
        if vision_ok:
            # Use vision comparison
            response = llm.invoke(messages=[{"role": "user", "content": content}])
        else:
            # Fallback to text-only comparison
            ...
    """
    content = []
    
    # Default comparison question
    if not comparison_question:
        comparison_question = f"""Compare the paper figure ({figure_id}) with the reproduction.

Please analyze:
1. Do both plots show the same general shape/trend?
2. Are the number of peaks/dips the same?
3. Do the peak positions appear at similar x-axis locations?
4. Are the relative amplitudes between features similar?
5. Is the overall shape (dip vs peak, symmetric vs asymmetric) the same?

Based on your analysis, classify the comparison as:
- SUCCESS: Main features match well
- PARTIAL: Some features match, some differences
- FAILURE: Major mismatch in key features"""
    
    # Try to encode paper figure
    paper_img = encode_image_for_claude(paper_figure_path)
    repro_img = encode_image_for_claude(reproduction_path)
    
    vision_available = bool(paper_img and repro_img)
    
    if vision_available:
        # Full vision comparison
        content.append({"type": "text", "text": f"**Paper Figure ({figure_id}):**"})
        content.append(paper_img)
        content.append({"type": "text", "text": "**Reproduction:**"})
        content.append(repro_img)
        content.append({"type": "text", "text": comparison_question})
    else:
        # Text-only fallback
        missing = []
        if not paper_img:
            missing.append(f"paper figure ({paper_figure_path})")
        if not repro_img:
            missing.append(f"reproduction ({reproduction_path})")
        
        fallback_text = f"""**Vision comparison unavailable** - Could not load: {', '.join(missing)}

Falling back to text-only comparison for {figure_id}.

Please compare based on available numerical data and descriptions only.
Mark this comparison as having lower confidence due to missing visual verification."""
        
        content.append({"type": "text", "text": fallback_text})
    
    return content, vision_available


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


def load_output_schema(agent_name: str) -> Dict[str, Any]:
    """
    Load an agent's output schema for function calling.
    
    This loads the JSON schema that defines the agent's output format.
    Use this schema with LLM function calling APIs to ensure structured,
    schema-compliant outputs.
    
    Args:
        agent_name: Short name (e.g., "plan_reviewer", "supervisor")
        
    Returns:
        JSON schema dict for function calling
        
    Raises:
        ValueError: If no output schema defined for this agent
        FileNotFoundError: If schema file doesn't exist
        
    Example:
        schema = load_output_schema("plan_reviewer")
        response = llm.invoke(
            prompt,
            tools=[{"type": "function", "function": {"name": "output", "parameters": schema}}]
        )
    """
    if agent_name not in AGENT_OUTPUT_SCHEMAS:
        raise ValueError(
            f"No output schema for agent: {agent_name}. "
            f"Available: {list(AGENT_OUTPUT_SCHEMAS.keys())}"
        )
    
    schema_filename = AGENT_OUTPUT_SCHEMAS[agent_name]
    schema_path = SCHEMAS_DIR / schema_filename
    
    if not schema_path.exists():
        raise FileNotFoundError(f"Output schema not found: {schema_path}")
    
    return json.loads(schema_path.read_text(encoding='utf-8'))


# ═══════════════════════════════════════════════════════════════════════
# Placeholder Injection
# ═══════════════════════════════════════════════════════════════════════

def inject_constants(template: str) -> str:
    """
    Inject constant values from state.py into a prompt template.
    
    Replaces placeholders like {THRESHOLDS_TABLE} with actual values
    generated from the canonical source in state.py.
    
    Supported placeholders:
    - {THRESHOLDS_TABLE}: Markdown table of discrepancy thresholds
    - {MAX_DESIGN_REVISIONS}: Maximum design revision attempts
    - {MAX_CODE_REVISIONS}: Maximum code revision attempts per stage
    - {MAX_ANALYSIS_REVISIONS}: Maximum analysis revision attempts
    - {MAX_REPLANS}: Maximum full replan attempts
    - {MAX_BACKTRACKS}: Default max backtracks (note: configurable via RuntimeConfig)
    
    Args:
        template: Prompt template with placeholders
        
    Returns:
        Template with placeholders replaced
    """
    # Import additional constants for injection
    from schemas.state import DEFAULT_RUNTIME_CONFIG
    
    # Define all constant placeholders and their values
    constants = {
        "{THRESHOLDS_TABLE}": format_thresholds_table(),
        "{MAX_DESIGN_REVISIONS}": str(MAX_DESIGN_REVISIONS),
        "{MAX_CODE_REVISIONS}": str(MAX_CODE_REVISIONS),
        "{MAX_ANALYSIS_REVISIONS}": str(MAX_ANALYSIS_REVISIONS),
        "{MAX_REPLANS}": str(MAX_REPLANS),
        "{MAX_BACKTRACKS}": str(DEFAULT_RUNTIME_CONFIG.get("max_backtracks", 2)),
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
    elif agent_name == "code_reviewer":
        context_section = _build_reviewer_context(state)
    elif agent_name == "execution_validator":
        context_section = _build_execution_context(state)
    elif agent_name == "physics_sanity":
        context_section = _build_physics_context(state)
    elif agent_name == "results_analyzer":
        context_section = _build_analyzer_context(state)
    elif agent_name == "comparison_validator":
        context_section = _build_comparison_validator_context(state)
    elif agent_name == "supervisor":
        context_section = _build_supervisor_context(state)
    elif agent_name == "prompt_adaptor":
        context_section = _build_adaptor_context(state)
    
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


def _extract_unit_system(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract unit_system from design JSON or performance estimate."""
    # Try to get from performance_estimate (which may contain parsed design)
    perf = state.get('performance_estimate', {})
    if isinstance(perf, dict) and 'unit_system' in perf:
        return perf['unit_system']
    
    # Try to parse from design_description if it's JSON
    design = state.get('design_description', '')
    if isinstance(design, str) and '"unit_system"' in design:
        try:
            import re
            # Try to extract unit_system block
            match = re.search(r'"unit_system"\s*:\s*\{[^}]+\}', design)
            if match:
                # This is a rough extraction - in production, parse the full JSON
                return {"note": "unit_system found in design - parse the full design JSON"}
        except:
            pass
    
    return {}


def _build_generator_context(state: Dict[str, Any]) -> str:
    """Build context section for CodeGeneratorAgent."""
    unit_system = _extract_unit_system(state)
    unit_system_text = json.dumps(unit_system, indent=2) if unit_system else "Not found in state - CHECK DESIGN JSON"
    
    return f"""
═══════════════════════════════════════════════════════════════════════
CURRENT TASK (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### ⚠️ UNIT SYSTEM (CRITICAL - USE THESE VALUES)
{unit_system_text}

**You MUST use the characteristic_length_m value for a_unit in your code.**
If unit_system is not shown above, extract it from the design JSON below.

### Design to Implement:

{state.get('design_description', '[No design provided]')}

### Previous Feedback (if revising):
{state.get('reviewer_feedback', 'N/A - first code attempt')}

### Performance Estimate:
{json.dumps(state.get('performance_estimate', {}), indent=2) if state.get('performance_estimate') else 'N/A'}

### Output Configuration:
- **Paper ID**: {state.get('paper_id', 'unknown')}
- **Stage ID**: {state.get('current_stage_id', 'unknown')}
"""


def _build_reviewer_context(state: Dict[str, Any]) -> str:
    """Build context section for CodeReviewerAgent."""
    # Determine if reviewing design or code
    has_code = bool(state.get('code'))
    has_design = bool(state.get('design_description'))
    review_type = "code" if has_code else "design" if has_design else "unknown"
    
    stage_info = _get_current_stage_info(state)
    
    context = f"""
═══════════════════════════════════════════════════════════════════════
REVIEW TASK (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Review Type: {review_type.upper()} REVIEW

### Stage Information:
- **Stage ID**: {state.get('current_stage_id', 'unknown')}
- **Stage Type**: {state.get('current_stage_type', 'unknown')}
- **Domain**: {state.get('paper_domain', 'other')}

### Stage Requirements:
{stage_info}
"""
    
    if review_type == "design":
        context += f"""
### Design to Review:

{state.get('design_description', '[No design provided]')}

### Performance Estimate:
{json.dumps(state.get('performance_estimate', {}), indent=2) if state.get('performance_estimate') else 'N/A'}
"""
    elif review_type == "code":
        code = state.get('code', '')
        unit_system = _extract_unit_system(state)
        unit_system_text = json.dumps(unit_system, indent=2) if unit_system else "Extract from design below"
        
        context += f"""
### ⚠️ UNIT SYSTEM (VERIFY a_unit MATCHES THIS)
{unit_system_text}

**BLOCKING CHECK**: Verify that the code's `a_unit` value matches 
`design["unit_system"]["characteristic_length_m"]`

### Code to Review:

```python
{code}
```

### What the Code Should Implement:
{state.get('design_description', '[Design not available]')[:500]}{'...' if len(state.get('design_description', '')) > 500 else ''}
"""
    
    return context


def _build_execution_context(state: Dict[str, Any]) -> str:
    """Build context section for ExecutionValidatorAgent."""
    outputs = state.get("stage_outputs", {})
    
    return f"""
═══════════════════════════════════════════════════════════════════════
EXECUTION VALIDATION TASK (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Execution Results:

- **Exit Code**: {outputs.get('exit_code', 'unknown')}
- **Runtime**: {outputs.get('runtime_seconds', 0):.1f} seconds
- **Stage ID**: {state.get('current_stage_id', 'unknown')}

### Output Files Created:
{chr(10).join('- ' + f for f in outputs.get('files', [])) or 'None'}

### Expected Outputs (from design):
{json.dumps(state.get('performance_estimate', {}).get('expected_outputs', []), indent=2) if state.get('performance_estimate') else 'Not specified'}

### Stdout:
```
{_truncate_output(outputs.get('stdout', ''), 100)}
```

### Stderr:
```
{outputs.get('stderr', '[no stderr]')}
```
"""


def _build_physics_context(state: Dict[str, Any]) -> str:
    """Build context section for PhysicsSanityAgent."""
    outputs = state.get("stage_outputs", {})
    
    # Get data file names
    data_files = [f for f in outputs.get('files', []) if f.endswith(('.csv', '.h5', '.npy'))]
    
    return f"""
═══════════════════════════════════════════════════════════════════════
PHYSICS VALIDATION TASK (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Stage Information:
- **Stage ID**: {state.get('current_stage_id', 'unknown')}
- **Stage Type**: {state.get('current_stage_type', 'unknown')}
- **Domain**: {state.get('paper_domain', 'other')}

### Data Files to Validate:
{chr(10).join('- ' + f for f in data_files) or 'No data files found'}

### Simulation Code (for context):
```python
{state.get('code', '[code not available]')[:2000]}{'...' if len(state.get('code', '')) > 2000 else ''}
```

### What Physics to Expect:
{_get_current_stage_info(state)}
"""


def _build_comparison_validator_context(state: Dict[str, Any]) -> str:
    """Build context section for ComparisonValidatorAgent."""
    comparisons = state.get("figure_comparisons", [])
    
    # Format recent comparisons for review
    comparisons_text = ""
    if comparisons:
        for comp in comparisons[-3:]:  # Last 3 comparisons
            comparisons_text += f"""
**{comp.get('figure_id', 'unknown')}**:
- Classification: {comp.get('classification', 'unknown')}
- Confidence: {comp.get('confidence', 0):.0%}
- Comparison Table: {json.dumps(comp.get('comparison_table', []), indent=2)}
"""
    else:
        comparisons_text = "No comparisons to validate"
    
    return f"""
═══════════════════════════════════════════════════════════════════════
COMPARISON VALIDATION TASK (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Analysis Summary:
{state.get('analysis_summary', '[No analysis summary available]')}

### Figure Comparisons to Validate:
{comparisons_text}

### Target Figures (reference):
{_get_target_figures_for_stage(state)}
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
    # Get validation hierarchy computed from progress (single source of truth)
    validation_hierarchy = get_validation_hierarchy(state)
    
    return f"""
═══════════════════════════════════════════════════════════════════════
CURRENT STATE (injected at runtime)
═══════════════════════════════════════════════════════════════════════

### Overall Progress:
{progress_summary}

### Validation Hierarchy Status:
{json.dumps(validation_hierarchy, indent=2)}

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

