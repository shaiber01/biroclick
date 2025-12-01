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
    # Status values (matches progress_schema.json):
    # - not_started: Stage hasn't been attempted
    # - in_progress: Stage is currently executing
    # - completed_success: Stage completed with good results
    # - completed_partial: Stage completed with partial match
    # - completed_failed: Stage completed but failed validation
    # - blocked: Stage skipped due to budget/dependencies
    # - needs_rerun: Stage needs to be re-executed (backtrack target)
    # - invalidated: Stage results invalid, will re-run when deps ready
    status: str
    last_updated: str  # ISO 8601
    revision_count: int
    runtime_seconds: NotRequired[float]
    summary: str
    outputs: List[Output]
    discrepancies: List[Discrepancy]
    issues: List[str]
    next_actions: List[str]
    # Backtracking support (matches progress_schema.json)
    invalidation_reason: NotRequired[str]  # Why this stage was invalidated (if status is invalidated/needs_rerun)
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
    paper_domain: str  # plasmonics | photonic_crystal | metamaterial | thin_film | waveguide | strong_coupling | nonlinear | other
    paper_text: str  # Full extracted text from PDF
    paper_title: str
    
    # ─── Runtime & Hardware Configuration ────────────────────────────────
    # These control execution behavior, timeouts, and resource limits.
    # Passed in via create_initial_state() or use defaults.
    runtime_config: "RuntimeConfig"  # Timeouts, debug mode, retry limits
    hardware_config: "HardwareConfig"  # CPU cores, RAM, GPU availability
    
    # ─── Shared Artifacts (mirrors of JSON files) ───────────────────────
    # These are kept in memory and periodically saved to disk
    plan: Dict[str, Any]  # Full plan structure
    assumptions: Dict[str, Any]  # Full assumptions structure
    progress: Dict[str, Any]  # Full progress structure
    
    # ─── Extracted Parameters ───────────────────────────────────────────
    # NOTE: Data Ownership Contract
    # 
    # The extracted_parameters field follows a specific sync pattern:
    #
    # 1. CANONICAL SOURCE: plan["extracted_parameters"]
    #    - Persisted to disk in plan_{paper_id}.json
    #    - PlannerAgent is the ONLY writer
    #    - Updated during PLAN node execution
    #
    # 2. IN-MEMORY VIEW: state.extracted_parameters
    #    - Typed in-memory copy for type safety
    #    - All agents (except Planner) READ from this field
    #    - Do NOT modify directly; changes won't persist
    #
    # 3. SYNC TIMING:
    #    - After PLAN node completes → sync from plan to state
    #    - At each checkpoint save → sync from plan to state
    #    - On checkpoint load → sync from loaded plan to state
    #
    # 4. SYNC IMPLEMENTATION:
    #    - Use sync_extracted_parameters(state) helper function
    #    - Performed automatically by workflow runner at sync points
    #
    # See sync_extracted_parameters() function below for implementation.
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
    code_revision_count: int  # Tracks code generation revisions
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
    
    # ─── Prompt Adaptations (from PromptAdaptorAgent) ───────────────────────
    # Stores the modifications made by PromptAdaptorAgent for this paper
    prompt_adaptations: List[Dict[str, Any]]  # List of {target_agent, modification_type, content, ...}


# ═══════════════════════════════════════════════════════════════════════
# State Initialization
# ═══════════════════════════════════════════════════════════════════════

def create_initial_state(
    paper_id: str,
    paper_text: str,
    paper_domain: str = "other",
    runtime_budget_minutes: float = 120.0,
    hardware_config: Optional[HardwareConfig] = None,
    runtime_config: Optional[RuntimeConfig] = None
) -> ReproState:
    """
    Create initial state for a new paper reproduction.
    
    Args:
        paper_id: Unique identifier for the paper
        paper_text: Extracted text from the paper PDF
        paper_domain: Primary domain (plasmonics, photonic_crystal, etc.)
        runtime_budget_minutes: Total runtime budget in minutes
        hardware_config: Optional hardware configuration (CPU cores, RAM, etc.)
                        Defaults to DEFAULT_HARDWARE_CONFIG if not provided.
        runtime_config: Optional runtime configuration (timeouts, debug mode, etc.)
                       Defaults to DEFAULT_RUNTIME_CONFIG if not provided.
    
    Returns:
        Initialized ReproState ready for the planning phase
    
    Example:
        # Basic usage with defaults
        state = create_initial_state("paper_123", paper_text)
        
        # With custom hardware config
        state = create_initial_state(
            "paper_123", 
            paper_text,
            hardware_config=HardwareConfig(cpu_cores=16, ram_gb=64, gpu_available=True)
        )
        
        # Debug mode
        state = create_initial_state(
            "paper_123",
            paper_text, 
            runtime_config=DEBUG_RUNTIME_CONFIG
        )
    """
    # Use defaults if not provided
    hw_config = hardware_config if hardware_config is not None else DEFAULT_HARDWARE_CONFIG
    rt_config = runtime_config if runtime_config is not None else DEFAULT_RUNTIME_CONFIG
    
    return ReproState(
        # Paper identification
        paper_id=paper_id,
        paper_domain=paper_domain,
        paper_text=paper_text,
        paper_title="",
        
        # Runtime & hardware configuration
        runtime_config=rt_config,
        hardware_config=hw_config,
        
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
        code_revision_count=0,
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
        paper_figures=[],
        
        # Prompt adaptations (populated by PromptAdaptorAgent)
        prompt_adaptations=[]
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
    
    # Debug mode settings
    debug_mode: bool  # Default: False - enables diagnostic quick-check mode
    debug_resolution_factor: float  # Default: 0.5 - multiplier for resolution in debug mode
    debug_max_stages: int  # Default: 2 - max stages to run in debug mode (Stage 0 + Stage 1)
    
    # Backtracking limits (moved from constants for configurability)
    max_backtracks: int  # Default: 2 - limit total backtracks to prevent infinite loops


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
    llm_retry_max_seconds=16.0,
    debug_mode=False,
    debug_resolution_factor=0.5,
    debug_max_stages=2,
    max_backtracks=2
)

# Debug mode runtime configuration preset
DEBUG_RUNTIME_CONFIG = RuntimeConfig(
    max_total_runtime_hours=0.5,  # 30 minutes max
    max_stage_runtime_minutes=10.0,  # 10 minutes per stage
    user_response_timeout_hours=1.0,  # 1 hour timeout in debug
    physics_retry_limit=1,  # Fewer retries in debug mode
    llm_retry_limit=3,
    json_parse_retry_limit=2,
    consecutive_failure_limit=1,
    llm_retry_base_seconds=1.0,
    llm_retry_max_seconds=8.0,
    debug_mode=True,
    debug_resolution_factor=0.5,
    debug_max_stages=2,
    max_backtracks=1
)


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# Revision limits
MAX_DESIGN_REVISIONS = 3
MAX_CODE_REVISIONS = 3  # Code generation revisions per stage
MAX_ANALYSIS_REVISIONS = 2
MAX_REPLANS = 2
# Note: MAX_BACKTRACKS is now configurable via RuntimeConfig.max_backtracks
# This constant is kept for backwards compatibility but prefer using RuntimeConfig
MAX_BACKTRACKS = 2  # Default limit; configurable via RuntimeConfig

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

# Context window limits (for paper length validation)
# Claude Opus 4.5 has 200K token context window
CONTEXT_WINDOW_LIMITS = {
    "model_max_tokens": 200000,  # Claude Opus 4.5 limit
    "system_prompt_reserve": 5000,  # Reserved for system prompts
    "state_context_reserve": 3000,  # Reserved for state context
    "response_reserve": 8000,  # Reserved for model response
    "safe_paper_tokens": 150000,  # Safe limit for paper text
    "chars_per_token_estimate": 4,  # Rough estimate for character counting
    "max_paper_chars": 600000,  # Hard limit for v1 (exits with error if exceeded)
}

# Paper length warning thresholds (in characters)
PAPER_LENGTH_THRESHOLDS = {
    "normal_max_chars": 50000,  # < 50K chars = normal
    "long_max_chars": 150000,  # 50K-150K chars = long (consider trimming)
    "very_long_max_chars": 600000,  # > 150K chars = very long (likely needs trimming)
}


# ═══════════════════════════════════════════════════════════════════════
# Context Window Loop Management
# ═══════════════════════════════════════════════════════════════════════
# 
# During revision loops (design→review→revise, code→review→revise), context
# can grow as feedback accumulates. These constants and functions help
# prevent context overflow.

LOOP_CONTEXT_LIMITS = {
    # Estimated tokens per revision loop component
    "design_description_avg_tokens": 2000,  # ~8K chars
    "code_avg_tokens": 3000,  # ~12K chars
    "reviewer_feedback_avg_tokens": 500,  # ~2K chars
    "analysis_summary_avg_tokens": 1500,  # ~6K chars
    
    # Context growth multiplier per loop iteration
    # Each iteration adds: previous feedback + new artifact
    "growth_factor_per_iteration": 1.3,  # 30% growth per loop
    
    # Safety thresholds
    "loop_context_warning_tokens": 15000,  # Warn if loop context exceeds this
    "loop_context_critical_tokens": 25000,  # Force summarization if exceeded
    
    # Maximum context to carry between loops
    "max_feedback_history_tokens": 3000,  # Truncate older feedback
    "max_iterations_with_full_context": 2,  # After this, summarize history
}


def estimate_loop_context_tokens(
    loop_type: str,
    iteration: int,
    current_artifact_chars: int = 0,
    feedback_history_chars: int = 0
) -> int:
    """
    Estimate total context tokens for a revision loop iteration.
    
    This helps detect when context is growing too large and summarization
    or truncation may be needed.
    
    Args:
        loop_type: "design", "code", or "analysis"
        iteration: Current iteration number (1-based)
        current_artifact_chars: Character count of current design/code/analysis
        feedback_history_chars: Character count of accumulated feedback
        
    Returns:
        Estimated total tokens for this loop iteration
        
    Example:
        >>> tokens = estimate_loop_context_tokens("code", iteration=2, 
        ...     current_artifact_chars=10000, feedback_history_chars=2000)
        >>> if tokens > LOOP_CONTEXT_LIMITS["loop_context_warning_tokens"]:
        ...     print("Consider summarizing feedback history")
    """
    limits = LOOP_CONTEXT_LIMITS
    chars_per_token = CONTEXT_WINDOW_LIMITS["chars_per_token_estimate"]
    
    # Base tokens for the artifact type
    base_tokens = {
        "design": limits["design_description_avg_tokens"],
        "code": limits["code_avg_tokens"],
        "analysis": limits["analysis_summary_avg_tokens"],
    }.get(loop_type, 2000)
    
    # Add actual artifact size if provided
    if current_artifact_chars > 0:
        artifact_tokens = current_artifact_chars // chars_per_token
    else:
        artifact_tokens = base_tokens
    
    # Feedback history tokens
    feedback_tokens = feedback_history_chars // chars_per_token
    
    # Growth factor for accumulated context
    growth = limits["growth_factor_per_iteration"] ** (iteration - 1)
    
    # Total estimate
    total = int((artifact_tokens + feedback_tokens) * growth)
    
    # Add base overhead (system prompt, state context)
    overhead = (
        CONTEXT_WINDOW_LIMITS["system_prompt_reserve"] +
        CONTEXT_WINDOW_LIMITS["state_context_reserve"]
    )
    
    return total + overhead


def check_loop_context_status(
    loop_type: str,
    iteration: int,
    current_artifact_chars: int = 0,
    feedback_history_chars: int = 0
) -> Dict[str, Any]:
    """
    Check if loop context is within safe limits and provide recommendations.
    
    Args:
        loop_type: "design", "code", or "analysis"
        iteration: Current iteration number (1-based)
        current_artifact_chars: Character count of current design/code/analysis
        feedback_history_chars: Character count of accumulated feedback
        
    Returns:
        Dict with:
        - status: "ok", "warning", or "critical"
        - estimated_tokens: Current estimated tokens
        - recommendation: Action to take (if any)
        - should_summarize: Whether to summarize feedback history
        
    Example:
        >>> status = check_loop_context_status("code", iteration=3,
        ...     current_artifact_chars=15000, feedback_history_chars=5000)
        >>> if status["should_summarize"]:
        ...     feedback = summarize_feedback_history(feedback_history)
    """
    limits = LOOP_CONTEXT_LIMITS
    
    estimated_tokens = estimate_loop_context_tokens(
        loop_type, iteration, current_artifact_chars, feedback_history_chars
    )
    
    result: Dict[str, Any] = {
        "estimated_tokens": estimated_tokens,
        "iteration": iteration,
        "loop_type": loop_type,
    }
    
    # Check against thresholds
    if estimated_tokens >= limits["loop_context_critical_tokens"]:
        result["status"] = "critical"
        result["recommendation"] = (
            f"Context critical ({estimated_tokens:,} tokens). "
            "Summarize feedback history before continuing. "
            "Consider keeping only most recent feedback."
        )
        result["should_summarize"] = True
        
    elif estimated_tokens >= limits["loop_context_warning_tokens"]:
        result["status"] = "warning"
        result["recommendation"] = (
            f"Context high ({estimated_tokens:,} tokens). "
            "Consider summarizing older feedback if more iterations needed."
        )
        result["should_summarize"] = iteration > limits["max_iterations_with_full_context"]
        
    else:
        result["status"] = "ok"
        result["recommendation"] = None
        result["should_summarize"] = False
    
    return result


def truncate_feedback_history(
    feedback_items: List[str],
    max_tokens: Optional[int] = None
) -> List[str]:
    """
    Truncate feedback history to stay within token limits.
    
    Keeps most recent feedback items, dropping oldest ones first.
    
    Args:
        feedback_items: List of feedback strings (oldest first)
        max_tokens: Maximum tokens to keep (default from LOOP_CONTEXT_LIMITS)
        
    Returns:
        Truncated list of feedback items (most recent preserved)
        
    Example:
        >>> history = ["Feedback 1...", "Feedback 2...", "Feedback 3..."]
        >>> truncated = truncate_feedback_history(history, max_tokens=1000)
    """
    if max_tokens is None:
        max_tokens = LOOP_CONTEXT_LIMITS["max_feedback_history_tokens"]
    
    chars_per_token = CONTEXT_WINDOW_LIMITS["chars_per_token_estimate"]
    max_chars = max_tokens * chars_per_token
    
    # Work backwards from most recent
    result = []
    total_chars = 0
    
    for feedback in reversed(feedback_items):
        feedback_chars = len(feedback)
        if total_chars + feedback_chars <= max_chars:
            result.insert(0, feedback)
            total_chars += feedback_chars
        else:
            # Can't fit any more
            break
    
    return result


def create_feedback_summary_prompt(feedback_items: List[str]) -> str:
    """
    Create a prompt for summarizing feedback history.
    
    When feedback history exceeds limits, this generates a prompt that can
    be sent to an LLM to create a condensed summary.
    
    Args:
        feedback_items: List of feedback strings to summarize
        
    Returns:
        Prompt string for LLM to generate summary
    """
    combined = "\n\n---\n\n".join(feedback_items)
    
    return f"""Summarize the following revision feedback history into a concise summary.
Keep only:
1. Unresolved issues that still need attention
2. Key constraints or requirements mentioned
3. Patterns in the feedback (recurring issues)

Drop:
- Issues that were already addressed
- Verbose explanations
- Redundant points

Feedback History:
{combined}

Provide a condensed summary (max 500 words):"""


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
    
    Also creates a "latest" pointer to this checkpoint for easy access.
    On Unix systems, the "latest" pointer is a symlink (space-efficient).
    On Windows or if symlink creation fails, falls back to a copy.
    
    Args:
        state: Current ReproState to save
        checkpoint_name: Name for this checkpoint (e.g., "after_plan", "stage2_complete")
        output_dir: Base output directory
        
    Returns:
        Path to saved checkpoint file
        
    Note:
        Windows symlink creation may fail without admin privileges or Developer Mode.
        The function gracefully falls back to file copy in such cases.
    """
    import json
    import os
    import shutil
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
    
    # Create "latest" pointer for easy access
    # Try symlink first (more space-efficient), fall back to copy
    latest_path = checkpoint_dir / f"checkpoint_{checkpoint_name}_latest.json"
    
    # Remove existing latest pointer (may be symlink or file)
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    
    symlink_created = False
    try:
        # Try to create a relative symlink (works better across systems)
        latest_path.symlink_to(filename)
        symlink_created = True
    except (OSError, NotImplementedError):
        # Symlink creation failed (common on Windows without admin/Developer Mode)
        # Fall back to file copy
        pass
    
    if not symlink_created:
        # Fall back: copy the checkpoint file
        shutil.copy2(filepath, latest_path)
    
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


# ═══════════════════════════════════════════════════════════════════════
# State Validation Guards
# ═══════════════════════════════════════════════════════════════════════

# Required state fields for each node
# This helps catch malformed state early in the workflow
NODE_REQUIREMENTS: Dict[str, List[str]] = {
    "ADAPT_PROMPTS": [
        "paper_id",
        "paper_text",
        "paper_domain",
    ],
    "PLAN": [
        "paper_id",
        "paper_text",
        "paper_domain",
        "paper_figures",
    ],
    "SELECT_STAGE": [
        "paper_id",
        "plan",
        "validation_hierarchy",
    ],
    "DESIGN": [
        "paper_id",
        "current_stage_id",
        "plan",
        "assumptions",
    ],
    "CODE_REVIEW_DESIGN": [
        "paper_id",
        "current_stage_id",
        "design_description",
        "plan",
    ],
    "GENERATE_CODE": [
        "paper_id",
        "current_stage_id",
        "design_description",
    ],
    "CODE_REVIEW_CODE": [
        "paper_id",
        "current_stage_id",
        "code",
        "design_description",
    ],
    "RUN_CODE": [
        "paper_id",
        "current_stage_id",
        "code",
    ],
    "EXECUTION_CHECK": [
        "paper_id",
        "current_stage_id",
        "stage_outputs",
    ],
    "PHYSICS_CHECK": [
        "paper_id",
        "current_stage_id",
        "stage_outputs",
        "code",
    ],
    "ANALYZE": [
        "paper_id",
        "current_stage_id",
        "stage_outputs",
        "paper_figures",
        "plan",
    ],
    "COMPARISON_CHECK": [
        "paper_id",
        "current_stage_id",
        "figure_comparisons",
        "analysis_summary",
    ],
    "SUPERVISOR": [
        "paper_id",
        "plan",
        "progress",
        "validation_hierarchy",
    ],
    "HANDLE_BACKTRACK": [
        "paper_id",
        "backtrack_decision",
        "progress",
    ],
    "ASK_USER": [
        "paper_id",
        "pending_user_questions",
    ],
    "GENERATE_REPORT": [
        "paper_id",
        "plan",
        "progress",
        "figure_comparisons",
        "assumptions",
    ],
}


def validate_state_for_node(state: ReproState, node_name: str) -> List[str]:
    """
    Check that state has all required fields for a specific node.
    
    This helps catch malformed state early, preventing cryptic errors
    from missing fields deep in agent logic.
    
    Args:
        state: Current ReproState to validate
        node_name: Name of the node about to execute
        
    Returns:
        List of missing field names (empty if all required fields present)
        
    Example:
        >>> missing = validate_state_for_node(state, "ANALYZE")
        >>> if missing:
        ...     raise ValueError(f"State missing required fields for ANALYZE: {missing}")
    """
    if node_name not in NODE_REQUIREMENTS:
        # Unknown node - no validation defined
        return []
    
    required = NODE_REQUIREMENTS[node_name]
    missing = []
    
    for field in required:
        if field not in state:
            missing.append(field)
        elif state.get(field) is None:
            # Field exists but is None - may be acceptable for some fields
            # Only flag as missing if it's a critical field
            if field in ["paper_id", "paper_text", "plan", "code", "stage_outputs"]:
                missing.append(f"{field} (is None)")
    
    return missing


def validate_state_transition(
    state: ReproState, 
    from_node: str, 
    to_node: str
) -> List[str]:
    """
    Validate that state is ready for transition between nodes.
    
    Checks both that required fields exist and that expected outputs
    from the previous node are present.
    
    Args:
        state: Current ReproState
        from_node: Node that just completed
        to_node: Node about to execute
        
    Returns:
        List of validation issues (empty if transition is valid)
    """
    issues = []
    
    # Check required fields for target node
    missing = validate_state_for_node(state, to_node)
    if missing:
        issues.extend([f"Missing field for {to_node}: {f}" for f in missing])
    
    # Check specific transition requirements
    transition_checks = {
        ("DESIGN", "CODE_REVIEW_DESIGN"): [
            ("design_description", "Design description not set after DESIGN node"),
        ],
        ("GENERATE_CODE", "CODE_REVIEW_CODE"): [
            ("code", "Code not set after GENERATE_CODE node"),
        ],
        ("RUN_CODE", "EXECUTION_CHECK"): [
            ("stage_outputs", "Stage outputs not set after RUN_CODE node"),
        ],
        ("ANALYZE", "COMPARISON_CHECK"): [
            ("analysis_summary", "Analysis summary not set after ANALYZE node"),
        ],
        ("PLAN", "SELECT_STAGE"): [
            ("plan", "Plan not set after PLAN node"),
            ("assumptions", "Assumptions not set after PLAN node"),
        ],
    }
    
    transition_key = (from_node, to_node)
    if transition_key in transition_checks:
        for field, message in transition_checks[transition_key]:
            if field not in state or state.get(field) is None:
                issues.append(message)
    
    return issues


# ═══════════════════════════════════════════════════════════════════════
# Extracted Parameters Sync
# ═══════════════════════════════════════════════════════════════════════

def sync_extracted_parameters(state: ReproState) -> ReproState:
    """
    Synchronize extracted_parameters from plan to state.
    
    The canonical source for extracted parameters is plan["extracted_parameters"].
    This function copies that data to state.extracted_parameters for type-safe
    access by agents.
    
    Call this function:
    - After PLAN node completes
    - After loading a checkpoint
    - Before each checkpoint save (ensures consistency)
    
    Args:
        state: ReproState to update (modified in place)
        
    Returns:
        The same state object (for chaining)
        
    Example:
        # After planning completes
        state = sync_extracted_parameters(state)
        
        # Or in workflow runner
        if current_node == "PLAN":
            state = sync_extracted_parameters(state)
    """
    plan = state.get("plan", {})
    plan_params = plan.get("extracted_parameters", [])
    
    # Convert to typed list of ExtractedParameter
    typed_params: List[ExtractedParameter] = []
    
    for param in plan_params:
        if isinstance(param, dict):
            # Ensure required fields exist with defaults
            # Fields match ExtractedParameter TypedDict and plan_schema.json
            typed_param: ExtractedParameter = {
                "name": param.get("name", "unnamed"),
                "value": param.get("value"),
                "unit": param.get("unit", ""),
                "source": param.get("source", "inferred"),
                "location": param.get("location", ""),
                "cross_checked": param.get("cross_checked", False),
                "discrepancy_notes": param.get("discrepancy_notes"),
            }
            typed_params.append(typed_param)
    
    state["extracted_parameters"] = typed_params
    
    return state


def get_extracted_parameter(
    state: ReproState,
    name: str,
    default: Any = None
) -> Any:
    """
    Get a specific extracted parameter by name.
    
    Convenience function for looking up parameter values.
    
    Args:
        state: Current ReproState
        name: Parameter name to find
        default: Value to return if not found
        
    Returns:
        Parameter value, or default if not found
        
    Example:
        disk_diameter = get_extracted_parameter(state, "disk_diameter", default=75)
    """
    params = state.get("extracted_parameters", [])
    
    for param in params:
        if param.get("name") == name:
            return param.get("value", default)
    
    return default


def list_extracted_parameters(
    state: ReproState,
    cross_checked_only: bool = False,
    source_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List all extracted parameters, optionally filtered by cross-check status or source.
    
    Args:
        state: Current ReproState
        cross_checked_only: If True, only return parameters that were cross-checked
        source_filter: Only return params from this source type
                      ("text", "figure_caption", "figure_axis", "supplementary", "inferred")
                          
    Returns:
        List of parameter dicts with name, value, unit, source, cross_checked
        
    Example:
        # Get all cross-checked parameters
        confirmed = list_extracted_parameters(state, cross_checked_only=True)
        
        # Get parameters extracted from figures
        from_figures = list_extracted_parameters(state, source_filter="figure_axis")
    """
    params = state.get("extracted_parameters", [])
    
    result = []
    for p in params:
        # Apply filters
        if cross_checked_only and not p.get("cross_checked", False):
            continue
        if source_filter and p.get("source") != source_filter:
            continue
        
        result.append({
            "name": p.get("name"),
            "value": p.get("value"),
            "unit": p.get("unit"),
            "source": p.get("source"),
            "cross_checked": p.get("cross_checked", False),
            "location": p.get("location"),
        })
    
    return result

