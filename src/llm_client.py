"""
LLM Client for ReproLab Multi-Agent System

This module provides a centralized interface for making LLM calls with:
- Schema-based structured output (function calling)
- Multimodal support (text + images)
- Token tracking and metrics
- Error handling with retries

All agents should use call_agent() from this module rather than
making direct LLM calls.

Environment Setup:
    Create a .env file in the project root with:
        ANTHROPIC_API_KEY=your-key-here
    Or set the environment variable directly.
"""

import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

# Default model for all agents (Claude Opus 4.5)
DEFAULT_MODEL = "claude-opus-4-5"

# Schema directory
SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0


# ═══════════════════════════════════════════════════════════════════════
# Schema Loading
# ═══════════════════════════════════════════════════════════════════════

_schema_cache: Dict[str, dict] = {}
_schema_cache_lock = threading.Lock()


def load_schema(schema_name: str) -> dict:
    """
    Load a JSON schema from the schemas directory.
    
    Thread-safe: Uses a lock to ensure concurrent calls return the same cached object.
    
    Args:
        schema_name: Schema filename (with or without .json extension)
        
    Returns:
        Parsed JSON schema dictionary
        
    Raises:
        FileNotFoundError: If schema file doesn't exist
        TypeError: If schema_name is not a string
    """
    if not isinstance(schema_name, str):
        raise TypeError(f"schema_name must be a string, got {type(schema_name).__name__}")
    
    # Normalize cache key: always use .json extension for consistency
    cache_key = schema_name if schema_name.endswith(".json") else f"{schema_name}.json"
    
    # Fast path: check cache without lock (safe for reads)
    if cache_key in _schema_cache:
        return _schema_cache[cache_key]
    
    # Slow path: acquire lock for thread-safe cache population
    with _schema_cache_lock:
        # Double-check after acquiring lock (another thread may have populated it)
        if cache_key in _schema_cache:
            return _schema_cache[cache_key]
        
        # Normalize filename for file access
        if not schema_name.endswith(".json"):
            schema_name = f"{schema_name}.json"
        
        schema_path = SCHEMAS_DIR / schema_name
        
        # Check if file exists, handling OSError for very long filenames
        try:
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema not found: {schema_path}")
        except OSError as e:
            # Handle very long filenames or other filesystem errors
            raise FileNotFoundError(f"Schema not found: {schema_path}") from e
        
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        
        _schema_cache[cache_key] = schema
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
        TypeError: If agent_name is not a string
    """
    if not isinstance(agent_name, str):
        raise TypeError(f"agent_name must be a string, got {type(agent_name).__name__}")
    
    # Check for empty string
    if not agent_name.strip():
        raise ValueError(f"Unknown agent: '{agent_name}'. Agent name cannot be empty or whitespace.")
    
    # Special cases that don't follow the standard naming convention
    special_mapping = {
        "report": "report_schema",
    }
    
    # Check special cases first
    if agent_name in special_mapping:
        return load_schema(special_mapping[agent_name])
    
    # Auto-discover: try {agent_name}_output_schema.json
    auto_schema_name = f"{agent_name}_output_schema"
    expected_filename = f"{auto_schema_name}.json"
    auto_schema_path = SCHEMAS_DIR / expected_filename
    
    # Check if file exists with exact case match (case-sensitive check)
    # On case-insensitive filesystems, Path.exists() might return True even with wrong case
    # So we verify by checking the actual directory listing
    try:
        if auto_schema_path.exists():
            # Verify exact case match by checking directory listing
            # This ensures case sensitivity even on case-insensitive filesystems
            try:
                actual_files = [f.name for f in SCHEMAS_DIR.iterdir() if f.is_file()]
                if expected_filename in actual_files:
                    return load_schema(auto_schema_name)
            except OSError:
                # If directory listing fails, fall back to exists() check
                # This handles the case where exists() works correctly
                return load_schema(auto_schema_name)
    except OSError:
        # Handle filesystem errors (e.g., very long paths)
        pass
    
    raise ValueError(
        f"Unknown agent: {agent_name}. "
        f"Expected schema at: {auto_schema_path}"
    )


# ═══════════════════════════════════════════════════════════════════════
# LLM Client
# ═══════════════════════════════════════════════════════════════════════

_llm_client: Optional[ChatAnthropic] = None
_llm_client_model: Optional[str] = None


def get_llm_client(model: Optional[str] = None) -> ChatAnthropic:
    """
    Get or create the LLM client singleton.
    
    Args:
        model: Optional model override (defaults to Claude Opus 4.5)
        
    Returns:
        ChatAnthropic client instance
        
    Configuration:
        - model: Claude Opus 4.5 (most capable model for scientific reasoning)
        - max_tokens: 16384 (accommodates thinking budget + response)
        - temperature: 1.0 (required when extended thinking is enabled)
        - thinking: enabled with 10000 token budget for complex reasoning
        - timeout: 5 minutes for long generations
        
    Note:
        Extended thinking is enabled for better reasoning on complex scientific
        tasks. This requires tool_choice="auto" instead of forcing specific tools.
    """
    global _llm_client, _llm_client_model
    
    model = model or os.environ.get("REPROLAB_MODEL", DEFAULT_MODEL)
    
    # If singleton exists but model differs, reset and recreate
    if _llm_client is not None and _llm_client_model != model:
        _llm_client = None
        _llm_client_model = None
    
    if _llm_client is None:
        _llm_client = ChatAnthropic(
            model=model,
            max_tokens=16384,    # Accommodates thinking budget + response
            timeout=300.0,       # 5 minute timeout for long generations
            temperature=1.0,     # Required when extended thinking is enabled
            thinking={
                "type": "enabled",
                "budget_tokens": 10000,  # Token budget for internal reasoning
            },
        )
        _llm_client_model = model
    
    return _llm_client


def reset_llm_client():
    """Reset the LLM client (useful for testing)."""
    global _llm_client, _llm_client_model
    _llm_client = None
    _llm_client_model = None


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
    # Use tool_choice="auto" to be compatible with extended thinking mode
    tool_name = f"submit_{agent_name}_output"
    llm_with_tools = llm.bind_tools(
        tools=[{
            "name": tool_name,
            "description": f"Submit the {agent_name} agent's structured output. You MUST use this tool to return your response.",
            "input_schema": schema,
        }],
        tool_choice={"type": "auto"},  # Compatible with thinking mode
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
                # With tool_choice="auto", model may respond without using tool
                content = response.content
                if isinstance(content, list):
                    # Extract text from content blocks (may include thinking)
                    content = " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                
                # Try to find and parse JSON in the response
                try:
                    import re
                    # First try to find JSON within code blocks (last one takes precedence)
                    code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
                    code_blocks = re.findall(code_block_pattern, content)
                    if code_blocks:
                        for block in reversed(code_blocks):
                            try:
                                return json.loads(block)
                            except json.JSONDecodeError:
                                continue
                    
                    # Fallback: try parsing from each '{' position, starting from the end
                    # This handles cases where set notation like {1, 2, 3} appears before real JSON
                    brace_positions = [i for i, c in enumerate(content) if c == '{']
                    for pos in reversed(brace_positions):
                        try:
                            # Try to parse JSON starting from this position
                            candidate = content[pos:]
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except json.JSONDecodeError:
                            continue
                    raise ValueError("No JSON found")
                except (json.JSONDecodeError, ValueError):
                    raise ValueError(
                        f"Agent {agent_name} did not return structured output. "
                        f"Response: {content[:500] if content else '(empty)'}"
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
        state["metrics"] = {}
    
    if "agent_calls" not in state["metrics"]:
        state["metrics"]["agent_calls"] = []
    
    if "stage_metrics" not in state["metrics"]:
        state["metrics"]["stage_metrics"] = []
    
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
            if isinstance(fig, dict)  # Skip non-dict items gracefully
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
    stage_id = state.get("current_stage_id") or "unknown"
    parts.append(f"# CURRENT STAGE: {stage_id}")
    
    # Plan stage details
    plan = state.get("plan") or {}
    stages = plan.get("stages", []) if isinstance(plan, dict) else []
    # Ensure stages is a list before iterating
    if not isinstance(stages, list):
        stages = []
    current_stage = next((s for s in stages if isinstance(s, dict) and s.get("stage_id") == stage_id), None)
    
    if current_stage:
        parts.append(f"## Stage Details\n```json\n{json.dumps(current_stage, indent=2, ensure_ascii=False)}\n```")
    
    # Extracted parameters
    params = state.get("extracted_parameters") or []
    if params:
        parts.append(f"## Extracted Parameters\n```json\n{json.dumps(params[:20], indent=2, ensure_ascii=False)}\n```")
    
    # Assumptions
    assumptions = state.get("assumptions") or {}
    if assumptions and isinstance(assumptions, dict):
        parts.append(f"## Assumptions\n```json\n{json.dumps(assumptions, indent=2, ensure_ascii=False)}\n```")
    
    # Validated materials (for Stage 1+)
    materials = state.get("validated_materials") or []
    if materials:
        parts.append(f"## Validated Materials\n```json\n{json.dumps(materials, indent=2, ensure_ascii=False)}\n```")
    
    # Revision feedback if any
    feedback = state.get("reviewer_feedback") or ""
    if feedback:
        parts.append(f"## REVISION FEEDBACK\n\n{feedback}")
    
    return "\n\n".join(parts)


def build_user_content_for_code_generator(state: Dict[str, Any]) -> str:
    """
    Build user content for the code generator agent.
    """
    parts = []
    
    # Current stage
    stage_id = state.get("current_stage_id") or "unknown"
    parts.append(f"# CURRENT STAGE: {stage_id}")
    
    # Design specification
    design = state.get("design_description") or ""
    if design:
        if isinstance(design, dict):
            parts.append(f"## Design Specification\n```json\n{json.dumps(design, indent=2, ensure_ascii=False)}\n```")
        else:
            parts.append(f"## Design Specification\n\n{design}")
    
    # Validated materials
    materials = state.get("validated_materials") or []
    if materials:
        parts.append(f"## Validated Materials (use these paths!)\n```json\n{json.dumps(materials, indent=2, ensure_ascii=False)}\n```")
    
    # Revision feedback if any
    feedback = state.get("reviewer_feedback") or ""
    if feedback:
        parts.append(f"## REVISION FEEDBACK\n\n{feedback}")
    
    return "\n\n".join(parts)


def build_user_content_for_analyzer(state: Dict[str, Any]) -> str:
    """
    Build user content for the results analyzer agent.
    """
    parts = []
    
    # Current stage
    stage_id = state.get("current_stage_id") or "unknown"
    parts.append(f"# CURRENT STAGE: {stage_id}")
    
    # Stage outputs
    outputs = state.get("stage_outputs")
    if outputs is not None:  # Include even if empty dict
        parts.append(f"## Simulation Outputs\n```json\n{json.dumps(outputs, indent=2, default=str)}\n```")
    
    # Target figures for this stage
    plan = state.get("plan") or {}
    stages = plan.get("stages", []) if isinstance(plan, dict) else []
    current_stage = next((s for s in stages if s.get("stage_id") == stage_id), None)
    
    if current_stage:
        targets = current_stage.get("targets", [])
        if targets:  # Only include if targets list is not empty
            parts.append(f"## Target Figures: {', '.join(targets)}")
    
    # Revision feedback if any
    feedback = state.get("analysis_feedback") or ""
    if feedback:
        parts.append(f"## REVISION FEEDBACK\n\n{feedback}")
    
    return "\n\n".join(parts)


def get_images_for_analyzer(state: Dict[str, Any]) -> List[Path]:
    """
    Get image paths for the results analyzer agent (multimodal).
    
    Returns paths to both paper figures and simulation output plots.
    """
    images = []
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".webp"]
    
    # Paper figures
    paper_figures = state.get("paper_figures") or []
    if paper_figures:
        for fig in paper_figures:
            if not isinstance(fig, dict):
                continue
            path = fig.get("image_path")
            if path:
                path_obj = Path(path)
                if path_obj.exists() and path_obj.suffix.lower() in image_extensions:
                    images.append(path_obj)
    
    # Simulation output plots
    stage_outputs = state.get("stage_outputs") or {}
    output_files = stage_outputs.get("files") if isinstance(stage_outputs, dict) else None
    if output_files is None:
        output_files = []
    
    for file_path in output_files:
        # Handle different path formats
        if hasattr(file_path, '__class__') and file_path.__class__.__name__ == 'Path':
            # Check if it's a Path object (works with both real Path and mocked Path)
            path = file_path
        elif isinstance(file_path, str):
            path = Path(file_path)
        elif isinstance(file_path, dict):
            path = Path(file_path.get("path", ""))
        else:
            # Fallback: try to convert to Path
            path = Path(str(file_path))
        
        if path.exists() and path.suffix.lower() in image_extensions:
            images.append(path)
    
    return images
