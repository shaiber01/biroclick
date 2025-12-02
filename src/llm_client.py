"""
LLM Client for ReproLab Multi-Agent System

This module provides a centralized interface for making LLM calls with:
- Schema-based structured output (function calling)
- Multimodal support (text + images)
- Token tracking and metrics
- Error handling with retries

All agents should use call_agent() from this module rather than
making direct LLM calls.
"""

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

# Default model for all agents (Claude Opus 4.5)
DEFAULT_MODEL = "claude-opus-4-20250514"

# Schema directory
SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0


# ═══════════════════════════════════════════════════════════════════════
# Schema Loading
# ═══════════════════════════════════════════════════════════════════════

_schema_cache: Dict[str, dict] = {}


def load_schema(schema_name: str) -> dict:
    """
    Load a JSON schema from the schemas directory.
    
    Args:
        schema_name: Schema filename (with or without .json extension)
        
    Returns:
        Parsed JSON schema dictionary
        
    Raises:
        FileNotFoundError: If schema file doesn't exist
    """
    if schema_name in _schema_cache:
        return _schema_cache[schema_name]
    
    if not schema_name.endswith(".json"):
        schema_name = f"{schema_name}.json"
    
    schema_path = SCHEMAS_DIR / schema_name
    
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    
    _schema_cache[schema_name] = schema
    return schema


def get_agent_schema(agent_name: str) -> dict:
    """
    Get the output schema for a specific agent.
    
    Uses auto-discovery: looks for {agent_name}_output_schema.json in the schemas
    directory. Falls back to special-case mapping for exceptions (e.g., "report").
    
    Args:
        agent_name: Agent name (e.g., "planner", "simulation_designer")
        
    Returns:
        Agent's output schema
        
    Raises:
        ValueError: If no schema found for the agent
    """
    # Special cases that don't follow the standard naming convention
    special_mapping = {
        "report": "report_schema",
    }
    
    # Check special cases first
    if agent_name in special_mapping:
        return load_schema(special_mapping[agent_name])
    
    # Auto-discover: try {agent_name}_output_schema.json
    auto_schema_name = f"{agent_name}_output_schema"
    auto_schema_path = SCHEMAS_DIR / f"{auto_schema_name}.json"
    
    if auto_schema_path.exists():
        return load_schema(auto_schema_name)
    
    raise ValueError(
        f"Unknown agent: {agent_name}. "
        f"Expected schema at: {auto_schema_path}"
    )


# ═══════════════════════════════════════════════════════════════════════
# LLM Client
# ═══════════════════════════════════════════════════════════════════════

_llm_client: Optional[ChatAnthropic] = None


def get_llm_client(model: Optional[str] = None) -> ChatAnthropic:
    """
    Get or create the LLM client singleton.
    
    Args:
        model: Optional model override (defaults to Claude Opus 4.5)
        
    Returns:
        ChatAnthropic client instance
    """
    global _llm_client
    
    model = model or os.environ.get("REPROLAB_MODEL", DEFAULT_MODEL)
    
    if _llm_client is None:
        _llm_client = ChatAnthropic(
            model=model,
            max_tokens=8192,  # Max output tokens
            timeout=300.0,    # 5 minute timeout for long generations
        )
    
    return _llm_client


def reset_llm_client():
    """Reset the LLM client (useful for testing)."""
    global _llm_client
    _llm_client = None


# ═══════════════════════════════════════════════════════════════════════
# Image Encoding for Multimodal
# ═══════════════════════════════════════════════════════════════════════

def encode_image_to_base64(image_path: Union[str, Path]) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string
    """
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: Union[str, Path]) -> str:
    """
    Get the MIME type for an image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        MIME type string (e.g., "image/png")
    """
    suffix = Path(image_path).suffix.lower()
    
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    
    return media_types.get(suffix, "image/png")


def create_image_content(image_path: Union[str, Path], detail: str = "auto") -> dict:
    """
    Create image content block for multimodal messages.
    
    Args:
        image_path: Path to image file
        detail: Image detail level ("auto", "low", "high")
        
    Returns:
        Image content dictionary for LangChain message
    """
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{get_image_media_type(image_path)};base64,{encode_image_to_base64(image_path)}",
            "detail": detail,
        }
    }


# ═══════════════════════════════════════════════════════════════════════
# Main Agent Call Function
# ═══════════════════════════════════════════════════════════════════════

def call_agent(
    agent_name: str,
    system_prompt: str,
    user_content: Union[str, List[dict]],
    schema_name: Optional[str] = None,
    images: Optional[List[Union[str, Path]]] = None,
    model: Optional[str] = None,
    max_retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    """
    Call an agent with structured output via function calling.
    
    This is the main entry point for all LLM calls in ReproLab.
    Uses Claude's tool use feature to ensure schema-compliant output.
    
    Args:
        agent_name: Name of the agent (used for schema lookup and tool naming)
        system_prompt: System prompt built by build_agent_prompt()
        user_content: User message content (string or multimodal content list)
        schema_name: Optional schema name override (defaults to agent's schema)
        images: Optional list of image paths for multimodal input
        model: Optional model override
        max_retries: Number of retries on transient errors
        
    Returns:
        Parsed agent output dictionary conforming to the schema
        
    Raises:
        ValueError: If agent output doesn't conform to schema
        RuntimeError: If all retries fail
        
    Example:
        >>> output = call_agent(
        ...     agent_name="planner",
        ...     system_prompt=build_agent_prompt("planner", state),
        ...     user_content=f"Paper text: {state['paper_text']}"
        ... )
    """
    # Get schema
    schema = get_agent_schema(agent_name) if schema_name is None else load_schema(schema_name)
    
    # Get LLM client with structured output
    llm = get_llm_client(model)
    
    # Create tool-enabled LLM with the schema
    tool_name = f"submit_{agent_name}_output"
    llm_with_tools = llm.bind_tools(
        tools=[{
            "name": tool_name,
            "description": f"Submit the {agent_name} agent's structured output",
            "input_schema": schema,
        }],
        tool_choice={"type": "tool", "name": tool_name},
    )
    
    # Build messages
    messages = [SystemMessage(content=system_prompt)]
    
    # Handle multimodal content
    if images:
        content_parts = []
        
        # Add text content
        if isinstance(user_content, str):
            content_parts.append({"type": "text", "text": user_content})
        else:
            content_parts.extend(user_content)
        
        # Add images
        for image_path in images:
            content_parts.append(create_image_content(image_path))
        
        messages.append(HumanMessage(content=content_parts))
    else:
        messages.append(HumanMessage(content=user_content))
    
    # Call with retries
    last_error = None
    for attempt in range(max_retries):
        try:
            response = llm_with_tools.invoke(messages)
            
            # Extract tool call result
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                return tool_call["args"]
            else:
                # Fallback: try to parse response content as JSON
                # This shouldn't happen with tool_choice set, but handle gracefully
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Agent {agent_name} did not return structured output. "
                        f"Response: {response.content[:500]}"
                    )
                    
        except Exception as e:
            last_error = e
            
            # Don't retry on validation errors
            if isinstance(e, ValueError):
                raise
            
            # Retry on transient errors
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
                continue
    
    raise RuntimeError(
        f"Agent {agent_name} failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


def call_agent_with_metrics(
    agent_name: str,
    system_prompt: str,
    user_content: Union[str, List[dict]],
    state: Dict[str, Any],
    schema_name: Optional[str] = None,
    images: Optional[List[Union[str, Path]]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call an agent and update state metrics with token usage.
    
    This is a wrapper around call_agent() that tracks token usage
    in the state's metrics field.
    
    Args:
        agent_name: Name of the agent
        system_prompt: System prompt
        user_content: User message content
        state: ReproState to update with metrics
        schema_name: Optional schema name override
        images: Optional image paths for multimodal
        model: Optional model override
        
    Returns:
        Parsed agent output dictionary
    """
    start_time = time.time()
    
    try:
        result = call_agent(
            agent_name=agent_name,
            system_prompt=system_prompt,
            user_content=user_content,
            schema_name=schema_name,
            images=images,
            model=model,
        )
        
        # Record success metrics
        duration = time.time() - start_time
        _record_call_metrics(state, agent_name, duration, success=True)
        
        return result
        
    except Exception as e:
        # Record failure metrics
        duration = time.time() - start_time
        _record_call_metrics(state, agent_name, duration, success=False, error=str(e))
        raise


def _record_call_metrics(
    state: Dict[str, Any],
    agent_name: str,
    duration: float,
    success: bool,
    error: Optional[str] = None,
):
    """Record LLM call metrics to state."""
    if "metrics" not in state:
        state["metrics"] = {"agent_calls": [], "stage_metrics": []}
    
    if "agent_calls" not in state["metrics"]:
        state["metrics"]["agent_calls"] = []
    
    state["metrics"]["agent_calls"].append({
        "agent": agent_name,
        "duration_seconds": duration,
        "success": success,
        "error": error,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


# ═══════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════

def build_user_content_for_planner(state: Dict[str, Any]) -> str:
    """
    Build user content for the planner agent.
    
    Includes paper text and figure information.
    """
    parts = []
    
    # Paper text
    paper_text = state.get("paper_text", "")
    if paper_text:
        parts.append(f"# PAPER TEXT\n\n{paper_text}")
    
    # Figure information
    figures = state.get("paper_figures", [])
    if figures:
        fig_desc = "\n".join([
            f"- {fig.get('id', 'unknown')}: {fig.get('description', 'No description')}"
            for fig in figures
        ])
        parts.append(f"# FIGURES\n\n{fig_desc}")
    
    # Replan feedback if any
    feedback = state.get("planner_feedback", "")
    if feedback:
        parts.append(f"# REVISION FEEDBACK\n\n{feedback}")
    
    return "\n\n---\n\n".join(parts)


def build_user_content_for_designer(state: Dict[str, Any]) -> str:
    """
    Build user content for the simulation designer agent.
    """
    parts = []
    
    # Current stage info
    stage_id = state.get("current_stage_id", "unknown")
    parts.append(f"# CURRENT STAGE: {stage_id}")
    
    # Plan stage details
    plan = state.get("plan", {})
    stages = plan.get("stages", [])
    current_stage = next((s for s in stages if s.get("stage_id") == stage_id), None)
    
    if current_stage:
        parts.append(f"## Stage Details\n```json\n{json.dumps(current_stage, indent=2)}\n```")
    
    # Extracted parameters
    params = state.get("extracted_parameters", [])
    if params:
        parts.append(f"## Extracted Parameters\n```json\n{json.dumps(params[:20], indent=2)}\n```")
    
    # Assumptions
    assumptions = state.get("assumptions", {})
    if assumptions:
        parts.append(f"## Assumptions\n```json\n{json.dumps(assumptions, indent=2)}\n```")
    
    # Validated materials (for Stage 1+)
    materials = state.get("validated_materials", [])
    if materials:
        parts.append(f"## Validated Materials\n```json\n{json.dumps(materials, indent=2)}\n```")
    
    # Revision feedback if any
    feedback = state.get("reviewer_feedback", "")
    if feedback:
        parts.append(f"## REVISION FEEDBACK\n\n{feedback}")
    
    return "\n\n".join(parts)


def build_user_content_for_code_generator(state: Dict[str, Any]) -> str:
    """
    Build user content for the code generator agent.
    """
    parts = []
    
    # Current stage
    stage_id = state.get("current_stage_id", "unknown")
    parts.append(f"# CURRENT STAGE: {stage_id}")
    
    # Design specification
    design = state.get("design_description", "")
    if design:
        if isinstance(design, dict):
            parts.append(f"## Design Specification\n```json\n{json.dumps(design, indent=2)}\n```")
        else:
            parts.append(f"## Design Specification\n\n{design}")
    
    # Validated materials
    materials = state.get("validated_materials", [])
    if materials:
        parts.append(f"## Validated Materials (use these paths!)\n```json\n{json.dumps(materials, indent=2)}\n```")
    
    # Revision feedback if any
    feedback = state.get("reviewer_feedback", "")
    if feedback:
        parts.append(f"## REVISION FEEDBACK\n\n{feedback}")
    
    return "\n\n".join(parts)


def build_user_content_for_analyzer(state: Dict[str, Any]) -> str:
    """
    Build user content for the results analyzer agent.
    """
    parts = []
    
    # Current stage
    stage_id = state.get("current_stage_id", "unknown")
    parts.append(f"# CURRENT STAGE: {stage_id}")
    
    # Stage outputs
    outputs = state.get("stage_outputs", {})
    if outputs:
        parts.append(f"## Simulation Outputs\n```json\n{json.dumps(outputs, indent=2, default=str)}\n```")
    
    # Target figures for this stage
    plan = state.get("plan", {})
    stages = plan.get("stages", [])
    current_stage = next((s for s in stages if s.get("stage_id") == stage_id), None)
    
    if current_stage:
        targets = current_stage.get("targets", [])
        parts.append(f"## Target Figures: {', '.join(targets)}")
    
    # Revision feedback if any
    feedback = state.get("analysis_feedback", "")
    if feedback:
        parts.append(f"## REVISION FEEDBACK\n\n{feedback}")
    
    return "\n\n".join(parts)


def get_images_for_analyzer(state: Dict[str, Any]) -> List[Path]:
    """
    Get image paths for the results analyzer agent (multimodal).
    
    Returns paths to both paper figures and simulation output plots.
    """
    images = []
    
    # Paper figures
    paper_figures = state.get("paper_figures", [])
    for fig in paper_figures:
        path = fig.get("image_path")
        if path and Path(path).exists():
            images.append(Path(path))
    
    # Simulation output plots
    stage_outputs = state.get("stage_outputs", {})
    output_files = stage_outputs.get("files", [])
    
    for file_path in output_files:
        path = Path(file_path) if isinstance(file_path, str) else Path(file_path.get("path", ""))
        if path.exists() and path.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
            images.append(path)
    
    return images

