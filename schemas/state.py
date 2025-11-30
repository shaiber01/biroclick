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


class CriticIssue(TypedDict):
    """An issue identified by CriticAgent."""
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
    """Structured comparison of a reproduced figure to paper."""
    figure_id: str
    title: str
    reproduction_image_path: NotRequired[str]
    comparison_table: List[FigureComparisonRow]
    shape_comparison: List[ShapeComparisonRow]
    reason_for_difference: str


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
    
    # ─── Verdicts ───────────────────────────────────────────────────────
    last_critic_verdict: Optional[str]  # approve_to_run | approve_results | needs_revision
    critic_issues: List[CriticIssue]
    supervisor_verdict: Optional[str]  # ok_continue | replan_needed | change_priority | ask_user
    
    # ─── Stage Working Data ─────────────────────────────────────────────
    code: Optional[str]  # Current Python+Meep code
    design_description: Optional[str]  # Natural language design
    performance_estimate: Optional[Dict[str, Any]]
    stage_outputs: Dict[str, Any]  # Filenames, stdout, etc.
    run_error: Optional[str]  # Capture simulation failures
    analysis_summary: Optional[str]  # Per-result report JSON
    
    # ─── Agent Feedback ─────────────────────────────────────────────────
    critic_feedback: Optional[str]  # Last critic feedback for revision
    supervisor_feedback: Optional[str]  # Last supervisor feedback
    planner_feedback: Optional[str]  # Feedback for replanning
    
    # ─── Performance Tracking ───────────────────────────────────────────
    runtime_budget_remaining_seconds: float
    total_runtime_seconds: float
    stage_start_time: Optional[str]  # ISO 8601
    
    # ─── User Interaction ───────────────────────────────────────────────
    pending_user_questions: List[str]
    user_responses: Dict[str, str]  # question → response
    awaiting_user_input: bool
    
    # ─── Report Generation ──────────────────────────────────────────────
    figure_comparisons: List[FigureComparison]  # All figure comparisons
    overall_assessment: List[OverallAssessmentItem]  # Executive summary
    systematic_discrepancies_identified: List[SystematicDiscrepancy]
    report_conclusions: Optional[ReportConclusions]
    final_report_markdown: Optional[str]  # Generated REPRODUCTION_REPORT.md


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
        
        # Verdicts
        last_critic_verdict=None,
        critic_issues=[],
        supervisor_verdict=None,
        
        # Stage working data
        code=None,
        design_description=None,
        performance_estimate=None,
        stage_outputs={},
        run_error=None,
        analysis_summary=None,
        
        # Agent feedback
        critic_feedback=None,
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
        
        # Report generation
        figure_comparisons=[],
        overall_assessment=[],
        systematic_discrepancies_identified=[],
        report_conclusions=None,
        final_report_markdown=None
    )


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# Revision limits
MAX_DESIGN_REVISIONS = 3
MAX_ANALYSIS_REVISIONS = 2
MAX_REPLANS = 2

# Default runtime budgets (in minutes)
DEFAULT_STAGE_BUDGETS = {
    "MATERIAL_VALIDATION": 5,
    "SINGLE_STRUCTURE": 15,
    "ARRAY_SYSTEM": 30,
    "PARAMETER_SWEEP": 60,
    "COMPLEX_PHYSICS": 120
}

# Discrepancy thresholds
DISCREPANCY_THRESHOLDS = {
    "resonance_wavelength": {"excellent": 2, "acceptable": 5, "investigate": 10},
    "linewidth": {"excellent": 10, "acceptable": 30, "investigate": 50},
    "q_factor": {"excellent": 10, "acceptable": 30, "investigate": 50},
    "transmission": {"excellent": 5, "acceptable": 15, "investigate": 30},
    "reflection": {"excellent": 5, "acceptable": 15, "investigate": 30},
    "field_enhancement": {"excellent": 20, "acceptable": 50, "investigate": 100},
    "effective_index": {"excellent": 1, "acceptable": 3, "investigate": 5}
}

# Stage types in validation hierarchy order
STAGE_TYPE_ORDER = [
    "MATERIAL_VALIDATION",
    "SINGLE_STRUCTURE", 
    "ARRAY_SYSTEM",
    "PARAMETER_SWEEP",
    "COMPLEX_PHYSICS"
]

