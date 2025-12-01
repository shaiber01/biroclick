"""
LangGraph State Definition for Paper Reproduction System

This module defines the TypedDict state that flows through the LangGraph
state machine. State is persisted between nodes and can be checkpointed.
"""

from typing import TypedDict, Optional, List, Dict, Any
from typing_extensions import NotRequired
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════
# Supporting Type Definitions
# ═══════════════════════════════════════════════════════════════════════

class ExtractedParameter(TypedDict):
    """A parameter extracted from the paper with provenance."""
    name: str
    value: Any  # number, string, or list
    unit: str
    source: str  # text | figure_caption | figure_axis | supplementary | inferred
    location: str
    cross_checked: bool
    discrepancy_notes: Optional[str]


class Discrepancy(TypedDict):
    """A documented discrepancy between simulation and paper."""
    id: str
    figure: str
    quantity: str
    paper_value: str
    simulation_value: str
    difference_percent: float
    classification: str  # excellent | acceptable | investigate
    likely_cause: str
    action_taken: str
    blocking: bool


class Output(TypedDict):
    """An output file produced by a stage."""
    type: str  # data | plot | log
    filename: str
    description: str
    target_figure: Optional[str]
    result_status: Optional[str]  # success | partial | failure
    comparison_notes: NotRequired[str]


class StageProgress(TypedDict):
    """Progress tracking for a single stage."""
    stage_id: str
    status: str  # not_started | in_progress | completed_success | completed_partial | completed_failed | blocked
    last_updated: str  # ISO 8601
    revision_count: int
    runtime_seconds: NotRequired[float]
    summary: str
    outputs: List[Output]
    discrepancies: List[Discrepancy]
    issues: List[str]
    next_actions: List[str]
    # Confidence fields
    classification_confidence: NotRequired[float]  # 0.0 to 1.0
    confidence_factors: NotRequired[List[str]]  # What affected confidence


class ReviewerIssue(TypedDict):
    """An issue identified by CodeReviewerAgent or validation agents."""
    severity: str  # blocking | major | minor
    category: str  # geometry | material | source | numerical | analysis | documentation
    description: str
    suggested_fix: str
    reference: NotRequired[str]


class FigureComparisonRow(TypedDict):
    """A row in a figure comparison table."""
    feature: str
    paper: str
    reproduction: str
    status: str  # "✅ Match" | "⚠️ XX%" | "❌ Mismatch"


class ShapeComparisonRow(TypedDict):
    """A row in a shape comparison table."""
    aspect: str
    paper: str
    reproduction: str


class FigureComparison(TypedDict):
    """
    Structured comparison of a reproduced figure to paper.
    
    This captures both the paths to images (for visual comparison by vision models)
    and structured data about the comparison results.
    """
    figure_id: str
    stage_id: str  # Which stage produced this comparison
    title: str
    
    # Image paths for comparison
    paper_image_path: str  # Path to original figure from paper (from PaperInput)
    reproduction_image_path: NotRequired[str]  # Path to simulation output image
    
    # Structured comparison data
    comparison_table: List[FigureComparisonRow]  # Quantitative comparison table
    shape_comparison: List[ShapeComparisonRow]  # Shape/feature comparison
    reason_for_difference: str  # Explanation of observed differences
    
    # Classification (from ResultsAnalyzerAgent)
    classification: str  # "success" | "partial" | "failure"
    
    # Vision model assessment (qualitative)
    visual_similarity: NotRequired[str]  # "high" | "medium" | "low"
    features_matched: NotRequired[List[str]]  # List of matched features
    features_mismatched: NotRequired[List[str]]  # List of mismatched features
    
    # Confidence fields
    confidence: float  # 0.0 to 1.0
    confidence_reason: str  # Explanation of confidence level


class OverallAssessmentItem(TypedDict):
    """Item in executive summary overall assessment."""
    aspect: str
    status: str
    icon: str  # "✅" | "⚠️" | "❌"
    notes: NotRequired[str]


class SystematicDiscrepancy(TypedDict):
    """A systematic discrepancy affecting multiple figures."""
    name: str
    description: str
    origin: str
    affected_figures: List[str]


class ReportConclusions(TypedDict):
    """Conclusions section of the reproduction report."""
    main_physics_reproduced: bool
    key_findings: List[str]
    final_statement: str


class ValidationHierarchyStatus(TypedDict):
    """Status of validation hierarchy stages."""
    material_validation: str  # passed | failed | partial | not_done
    single_structure: str
    arrays_systems: str
    parameter_sweeps: str


class UserInteractionContext(TypedDict, total=False):
    """Context information for a user interaction."""
    stage_id: Optional[str]
    agent: str
    reason: str


class UserInteraction(TypedDict):
    """
    A logged user decision or feedback during reproduction.
    
    User interactions are important to track because:
    - They document key decisions that affected the reproduction
    - They provide context for why certain approaches were taken
    - They help in reproducing the reproduction (meta-reproducibility)
    - They inform future system improvements
    """
    id: str  # Unique ID (e.g., "U1", "U2")
    timestamp: str  # ISO 8601
    interaction_type: str  # material_checkpoint | clarification | trade_off_decision | parameter_confirmation | stop_decision | backtrack_approval | general_feedback
    context: UserInteractionContext
    question: str  # Question posed to user
    user_response: str  # User's response/decision
    impact: NotRequired[str]  # How this affected the reproduction
    alternatives_considered: NotRequired[List[str]]  # Other options presented


class AgentCallMetric(TypedDict):
    """Metrics for a single agent call."""
    agent: str
    node: str
    stage_id: Optional[str]
    timestamp: str  # ISO 8601
    duration_seconds: float
    input_tokens: NotRequired[int]
    output_tokens: NotRequired[int]
    model: NotRequired[str]
    verdict: NotRequired[str]
    error: NotRequired[str]


class StageMetric(TypedDict):
    """Metrics for a single stage."""
    stage_id: str
    stage_type: str
    started_at: Optional[str]
    completed_at: Optional[str]
    duration_seconds: NotRequired[float]
    simulation_runtime_seconds: NotRequired[float]
    design_revisions: int
    code_revisions: int
    analysis_revisions: int
    final_status: NotRequired[str]
    escalated_to_user: bool


class MetricsLog(TypedDict):
    """
    Minimal live metrics tracked during execution.
    
    Used for monitoring and future PromptEvolutionAgent learning.
    
    NOTE: This is intentionally simpler than metrics_schema.json.
    The full schema (with revision_summary, token_summary, reproduction_quality,
    etc.) is used for the final exported metrics file. That richer structure
    is computed from this live log at GENERATE_REPORT time.
    
    Live tracking (this TypedDict):
    - Basic counters and timestamps
    - Raw agent call metrics
    - Stage-level metrics
    
    Exported format (metrics_schema.json):
    - All of the above, plus
    - revision_summary (aggregated counts)
    - token_summary (cost analysis)
    - reproduction_quality (final assessment)
    """
    paper_id: str
    started_at: str  # ISO 8601
    completed_at: Optional[str]
    total_duration_seconds: NotRequired[float]
    final_status: str  # in_progress | completed | stopped_by_user | failed
    agent_calls: List[AgentCallMetric]
    stage_metrics: List[StageMetric]
    total_input_tokens: int
    total_output_tokens: int
    prompt_adaptations_count: int


# ═══════════════════════════════════════════════════════════════════════
# Main State Definition
# ═══════════════════════════════════════════════════════════════════════

class ReproState(TypedDict, total=False):
    """
    Main state object for the paper reproduction LangGraph.
    
    This state flows through all nodes and maintains the complete
    context of the reproduction effort.
    """
    
    # ─── Paper Identification ───────────────────────────────────────────
    paper_id: str
    paper_domain: str  # plasmonics | photonic_crystal | metamaterial | thin_film | other
    paper_text: str  # Full extracted text from PDF
    paper_title: str
    
    # ─── Shared Artifacts (mirrors of JSON files) ───────────────────────
    # These are kept in memory and periodically saved to disk
    plan: Dict[str, Any]  # Full plan structure
    assumptions: Dict[str, Any]  # Full assumptions structure
    progress: Dict[str, Any]  # Full progress structure
    
    # ─── Extracted Parameters ───────────────────────────────────────────
    # NOTE: Data Ownership Contract
    # - plan["extracted_parameters"] = canonical persisted structure (saved to disk)
    # - state.extracted_parameters = typed in-memory view, synced FROM plan
    # - PlannerAgent writes to plan["extracted_parameters"]
    # - Other agents READ from state.extracted_parameters
    # - Sync happens at checkpoint boundaries
    extracted_parameters: List[ExtractedParameter]
    
    # ─── Validation Tracking ────────────────────────────────────────────
    validation_hierarchy: ValidationHierarchyStatus
    geometry_interpretations: Dict[str, str]  # ambiguous_term → interpretation
    discrepancies_log: List[Discrepancy]  # All discrepancies across all stages
    systematic_shifts: List[str]  # Known systematic shifts
    
    # ─── Current Control ────────────────────────────────────────────────
    current_stage_id: Optional[str]
    current_stage_type: Optional[str]  # MATERIAL_VALIDATION | SINGLE_STRUCTURE | etc.
    workflow_phase: str  # planning | design | pre_run_review | running | analysis | post_run_review | supervision
    
    # ─── Revision Tracking ──────────────────────────────────────────────
    design_revision_count: int
    analysis_revision_count: int
    replan_count: int
    
    # ─── Backtracking Support ─────────────────────────────────────────────
    # When an agent detects a significant issue that invalidates earlier work,
    # it can suggest backtracking. SupervisorAgent decides whether to accept.
    backtrack_suggestion: Optional[Dict[str, Any]]  # {suggesting_agent, target_stage_id, reason, severity}
    invalidated_stages: List[str]  # Stage IDs marked as needing re-run
    backtrack_count: int  # Track number of backtracks (for limits)
    
    # ─── Verdicts ───────────────────────────────────────────────────────
    last_reviewer_verdict: Optional[str]  # approve_to_run | approve_results | needs_revision
    reviewer_issues: List[ReviewerIssue]
    supervisor_verdict: Optional[str]  # ok_continue | replan_needed | change_priority | ask_user | backtrack_to_stage
    backtrack_decision: Optional[Dict[str, Any]]  # {accepted, target_stage_id, stages_to_invalidate, reason}
    
    # ─── Stage Working Data ─────────────────────────────────────────────
    code: Optional[str]  # Current Python+Meep code
    design_description: Optional[str]  # Natural language design
    performance_estimate: Optional[Dict[str, Any]]
    stage_outputs: Dict[str, Any]  # Filenames, stdout, etc.
    run_error: Optional[str]  # Capture simulation failures
    analysis_summary: Optional[str]  # Per-result report JSON
    
    # ─── Agent Feedback ─────────────────────────────────────────────────
    reviewer_feedback: Optional[str]  # Last reviewer feedback for revision
    supervisor_feedback: Optional[str]  # Last supervisor feedback
    planner_feedback: Optional[str]  # Feedback for replanning
    
    # ─── Performance Tracking ───────────────────────────────────────────
    runtime_budget_remaining_seconds: float
    total_runtime_seconds: float
    stage_start_time: Optional[str]  # ISO 8601
    
    # ─── User Interaction ───────────────────────────────────────────────
    pending_user_questions: List[str]
    user_responses: Dict[str, str]  # question → response (current session)
    awaiting_user_input: bool
    user_interactions: List[UserInteraction]  # Full log of all user decisions/feedback
    
    # ─── Report Generation ──────────────────────────────────────────────
    figure_comparisons: List[FigureComparison]  # All figure comparisons
    overall_assessment: List[OverallAssessmentItem]  # Executive summary
    systematic_discrepancies_identified: List[SystematicDiscrepancy]
    report_conclusions: Optional[ReportConclusions]
    final_report_markdown: Optional[str]  # Generated REPRODUCTION_REPORT.md
    
    # ─── Metrics Tracking ─────────────────────────────────────────────────
    metrics: Optional[MetricsLog]  # Comprehensive metrics for monitoring and learning
    
    # ─── Paper Figures (for multimodal comparison) ────────────────────────
    paper_figures: List[Dict[str, str]]  # [{id, description, image_path}, ...]


# ═══════════════════════════════════════════════════════════════════════
# State Initialization
# ═══════════════════════════════════════════════════════════════════════

def create_initial_state(
    paper_id: str,
    paper_text: str,
    paper_domain: str = "other",
    runtime_budget_minutes: float = 120.0
) -> ReproState:
    """
    Create initial state for a new paper reproduction.
    
    Args:
        paper_id: Unique identifier for the paper
        paper_text: Extracted text from the paper PDF
        paper_domain: Primary domain (plasmonics, photonic_crystal, etc.)
        runtime_budget_minutes: Total runtime budget in minutes
    
    Returns:
        Initialized ReproState ready for the planning phase
    """
    return ReproState(
        # Paper identification
        paper_id=paper_id,
        paper_domain=paper_domain,
        paper_text=paper_text,
        paper_title="",
        
        # Shared artifacts (empty, to be filled by PlannerAgent)
        plan={},
        assumptions={},
        progress={},
        
        # Extracted parameters
        extracted_parameters=[],
        
        # Validation tracking
        validation_hierarchy={
            "material_validation": "not_done",
            "single_structure": "not_done",
            "arrays_systems": "not_done",
            "parameter_sweeps": "not_done"
        },
        geometry_interpretations={},
        discrepancies_log=[],
        systematic_shifts=[],
        
        # Current control
        current_stage_id=None,
        current_stage_type=None,
        workflow_phase="planning",
        
        # Revision tracking
        design_revision_count=0,
        analysis_revision_count=0,
        replan_count=0,
        
        # Backtracking support
        backtrack_suggestion=None,
        invalidated_stages=[],
        backtrack_count=0,
        
        # Verdicts
        last_reviewer_verdict=None,
        reviewer_issues=[],
        supervisor_verdict=None,
        backtrack_decision=None,
        
        # Stage working data
        code=None,
        design_description=None,
        performance_estimate=None,
        stage_outputs={},
        run_error=None,
        analysis_summary=None,
        
        # Agent feedback
        reviewer_feedback=None,
        supervisor_feedback=None,
        planner_feedback=None,
        
        # Performance tracking
        runtime_budget_remaining_seconds=runtime_budget_minutes * 60,
        total_runtime_seconds=0.0,
        stage_start_time=None,
        
        # User interaction
        pending_user_questions=[],
        user_responses={},
        awaiting_user_input=False,
        user_interactions=[],  # Log of all user decisions/feedback
        
        # Report generation
        figure_comparisons=[],
        overall_assessment=[],
        systematic_discrepancies_identified=[],
        report_conclusions=None,
        final_report_markdown=None,
        
        # Metrics tracking
        metrics={
            "paper_id": paper_id,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "final_status": "in_progress",
            "agent_calls": [],
            "stage_metrics": [],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "prompt_adaptations_count": 0
        },
        
        # Paper figures (populated from PaperInput)
        paper_figures=[]
    )


# ═══════════════════════════════════════════════════════════════════════
# Hardware Configuration
# ═══════════════════════════════════════════════════════════════════════

class HardwareConfig(TypedDict):
    """
    Hardware configuration for simulation execution.
    Used by SimulationDesignerAgent for runtime estimates and
    by CodeGeneratorAgent for parallelization decisions.
    """
    cpu_cores: int  # Number of CPU cores available
    ram_gb: int  # RAM in gigabytes
    gpu_available: bool  # Whether GPU acceleration is available (for future use)


# Default hardware configuration (power laptop)
DEFAULT_HARDWARE_CONFIG = HardwareConfig(
    cpu_cores=8,
    ram_gb=32,
    gpu_available=False
)


# ═══════════════════════════════════════════════════════════════════════
# Runtime Configuration
# ═══════════════════════════════════════════════════════════════════════

class RuntimeConfig(TypedDict):
    """
    Runtime configuration for the reproduction workflow.
    These values control timeouts, limits, and error recovery behavior.
    """
    # Total time budget
    max_total_runtime_hours: float  # Default: 8.0
    max_stage_runtime_minutes: float  # Default: 60.0
    
    # User interaction
    user_response_timeout_hours: float  # Default: 24.0
    
    # Error recovery limits
    physics_retry_limit: int  # Default: 2
    llm_retry_limit: int  # Default: 5
    json_parse_retry_limit: int  # Default: 3
    consecutive_failure_limit: int  # Default: 2
    
    # LLM retry backoff (exponential: 1s, 2s, 4s, 8s, 16s)
    llm_retry_base_seconds: float  # Default: 1.0
    llm_retry_max_seconds: float  # Default: 16.0


# Default runtime configuration
DEFAULT_RUNTIME_CONFIG = RuntimeConfig(
    max_total_runtime_hours=8.0,
    max_stage_runtime_minutes=60.0,
    user_response_timeout_hours=24.0,
    physics_retry_limit=2,
    llm_retry_limit=5,
    json_parse_retry_limit=3,
    consecutive_failure_limit=2,
    llm_retry_base_seconds=1.0,
    llm_retry_max_seconds=16.0
)


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# Revision limits
MAX_DESIGN_REVISIONS = 3
MAX_ANALYSIS_REVISIONS = 2
MAX_REPLANS = 2
MAX_BACKTRACKS = 2  # Limit total backtracks to prevent infinite loops

# Default runtime budgets (in minutes)
DEFAULT_STAGE_BUDGETS = {
    "MATERIAL_VALIDATION": 5,
    "SINGLE_STRUCTURE": 15,
    "ARRAY_SYSTEM": 30,
    "PARAMETER_SWEEP": 60,
    "COMPLEX_PHYSICS": 120
}

# Discrepancy thresholds (canonical source of truth)
# Values are percentages: 2 means ±2%
DISCREPANCY_THRESHOLDS = {
    "resonance_wavelength": {"excellent": 2, "acceptable": 5, "investigate": 10},
    "linewidth": {"excellent": 10, "acceptable": 30, "investigate": 50},
    "q_factor": {"excellent": 10, "acceptable": 30, "investigate": 50},
    "transmission": {"excellent": 5, "acceptable": 15, "investigate": 30},
    "reflection": {"excellent": 5, "acceptable": 15, "investigate": 30},
    "field_enhancement": {"excellent": 20, "acceptable": 50, "investigate": 100},
    "effective_index": {"excellent": 1, "acceptable": 3, "investigate": 5}
}


def format_thresholds_table() -> str:
    """
    Generate a markdown table from DISCREPANCY_THRESHOLDS.
    
    This function is used by the prompt builder to inject the canonical
    thresholds into agent prompts at runtime, ensuring single source of truth.
    
    Returns:
        Markdown table string with thresholds
    """
    # Human-readable names for quantities
    quantity_names = {
        "resonance_wavelength": "Resonance wavelength",
        "linewidth": "Linewidth / FWHM",
        "q_factor": "Q-factor",
        "transmission": "Transmission",
        "reflection": "Reflection",
        "field_enhancement": "Field enhancement",
        "effective_index": "Mode effective index",
    }
    
    rows = [
        "| Quantity | Excellent | Acceptable | Investigate |",
        "|----------|-----------|------------|-------------|"
    ]
    
    for key, thresholds in DISCREPANCY_THRESHOLDS.items():
        name = quantity_names.get(key, key.replace("_", " ").title())
        excellent = thresholds["excellent"]
        acceptable = thresholds["acceptable"]
        investigate = thresholds["investigate"]
        
        # Format: ±X% for excellent/acceptable, >X% for investigate
        rows.append(
            f"| {name} | ±{excellent}% | ±{acceptable}% | >{investigate}% |"
        )
    
    return "\n".join(rows)


def get_threshold_for_quantity(quantity: str, level: str = "acceptable") -> float:
    """
    Get the threshold value for a specific quantity and level.
    
    Args:
        quantity: Key from DISCREPANCY_THRESHOLDS (e.g., "resonance_wavelength")
        level: "excellent", "acceptable", or "investigate"
        
    Returns:
        Threshold as a decimal (e.g., 0.05 for 5%)
        
    Raises:
        KeyError: If quantity or level not found
    """
    return DISCREPANCY_THRESHOLDS[quantity][level] / 100.0

# Stage types in validation hierarchy order
STAGE_TYPE_ORDER = [
    "MATERIAL_VALIDATION",
    "SINGLE_STRUCTURE", 
    "ARRAY_SYSTEM",
    "PARAMETER_SWEEP",
    "COMPLEX_PHYSICS"
]


# ═══════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════

# Checkpoint locations in the workflow
CHECKPOINT_LOCATIONS = [
    "after_plan",           # After PlannerAgent completes
    "after_stage_complete", # After each stage completes (SUPERVISOR node)
    "before_ask_user",      # Before pausing for user input
]

# Checkpoint naming convention: checkpoint_<paper_id>_<location>_<timestamp>.json


def save_checkpoint(
    state: ReproState,
    checkpoint_name: str,
    output_dir: str = "outputs"
) -> str:
    """
    Save a checkpoint of the current state.
    
    Args:
        state: Current ReproState to save
        checkpoint_name: Name for this checkpoint (e.g., "after_plan", "stage2_complete")
        output_dir: Base output directory
        
    Returns:
        Path to saved checkpoint file
    """
    import json
    from pathlib import Path
    
    paper_id = state.get("paper_id", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    checkpoint_dir = Path(output_dir) / paper_id / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"checkpoint_{paper_id}_{checkpoint_name}_{timestamp}.json"
    filepath = checkpoint_dir / filename
    
    # Convert state to JSON-serializable dict
    state_dict = dict(state)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, indent=2, default=str)
    
    # Also save as "latest" for easy access
    latest_path = checkpoint_dir / f"checkpoint_{checkpoint_name}_latest.json"
    with open(latest_path, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, indent=2, default=str)
    
    return str(filepath)


def load_checkpoint(
    paper_id: str,
    checkpoint_name: str = "latest",
    output_dir: str = "outputs"
) -> Optional[ReproState]:
    """
    Load a checkpoint to resume a reproduction.
    
    Args:
        paper_id: Paper identifier
        checkpoint_name: Specific checkpoint name or "latest" for most recent
        output_dir: Base output directory
        
    Returns:
        Loaded ReproState or None if not found
    """
    import json
    from pathlib import Path
    
    checkpoint_dir = Path(output_dir) / paper_id / "checkpoints"
    
    if checkpoint_name == "latest":
        # Find most recent checkpoint
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return None
        # Sort by modification time, get most recent
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        filepath = latest
    else:
        # Look for specific checkpoint
        filepath = checkpoint_dir / f"checkpoint_{checkpoint_name}_latest.json"
        if not filepath.exists():
            # Try with timestamp pattern
            matches = list(checkpoint_dir.glob(f"checkpoint_{paper_id}_{checkpoint_name}_*.json"))
            if not matches:
                return None
            filepath = max(matches, key=lambda p: p.stat().st_mtime)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        state_dict = json.load(f)
    
    return state_dict


def list_checkpoints(paper_id: str, output_dir: str = "outputs") -> List[Dict[str, str]]:
    """
    List all available checkpoints for a paper.
    
    Args:
        paper_id: Paper identifier
        output_dir: Base output directory
        
    Returns:
        List of checkpoint info dicts with name, timestamp, path
    """
    from pathlib import Path
    
    checkpoint_dir = Path(output_dir) / paper_id / "checkpoints"
    
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    for cp_file in checkpoint_dir.glob("checkpoint_*.json"):
        if "_latest" in cp_file.name:
            continue  # Skip "latest" symlinks
        
        # Parse filename: checkpoint_<paper_id>_<name>_<timestamp>.json
        parts = cp_file.stem.split("_")
        if len(parts) >= 4:
            name = "_".join(parts[2:-2])  # Everything between paper_id and timestamp
            timestamp = "_".join(parts[-2:])  # Last two parts are timestamp
        else:
            name = cp_file.stem
            timestamp = "unknown"
        
        checkpoints.append({
            "name": name,
            "timestamp": timestamp,
            "path": str(cp_file),
            "size_kb": cp_file.stat().st_size / 1024
        })
    
    # Sort by timestamp (most recent first)
    checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return checkpoints

