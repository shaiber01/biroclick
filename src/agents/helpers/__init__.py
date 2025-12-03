"""
Helper modules for agent node implementations.

These modules contain pure utility functions extracted from agents.py
to improve maintainability and testability.
"""

from .context import (
    check_context_or_escalate,
    validate_user_responses,
    validate_state_or_warn,
)
from .stubs import (
    ensure_stub_figures,
    build_stub_targets,
    build_stub_expected_outputs,
    build_stub_stages,
    build_stub_planned_materials,
    build_stub_assumptions,
    build_stub_plan,
)
from .numeric import (
    resolve_data_path,
    normalize_series,
    units_to_multiplier,
    load_numeric_series,
    compute_peak_metrics,
    quantitative_curve_metrics,
)
from .validation import (
    classify_percent_error,
    classification_from_metrics,
    CRITERIA_PATTERNS,
    evaluate_validation_criteria,
    extract_targets_from_feedback,
    match_output_file,
    normalize_output_file_entry,
    collect_expected_outputs,
    collect_expected_columns,
    match_expected_files,
    stage_comparisons_for_stage,
    analysis_reports_for_stage,
    validate_analysis_reports,
    breakdown_comparison_classifications,
)
from .metrics import (
    log_agent_call,
    record_discrepancy,
)
from .materials import (
    materials_from_stage_outputs,
    extract_materials_from_plan_assumptions,
    deduplicate_materials,
    load_material_database,
    match_material_from_text,
    format_validated_material,
    extract_validated_materials,
    format_material_checkpoint_question,
)

__all__ = [
    # context
    "check_context_or_escalate",
    "validate_user_responses",
    "validate_state_or_warn",
    # stubs
    "ensure_stub_figures",
    "build_stub_targets",
    "build_stub_expected_outputs",
    "build_stub_stages",
    "build_stub_planned_materials",
    "build_stub_assumptions",
    "build_stub_plan",
    # numeric
    "resolve_data_path",
    "normalize_series",
    "units_to_multiplier",
    "load_numeric_series",
    "compute_peak_metrics",
    "quantitative_curve_metrics",
    # validation
    "classify_percent_error",
    "classification_from_metrics",
    "CRITERIA_PATTERNS",
    "evaluate_validation_criteria",
    "extract_targets_from_feedback",
    "match_output_file",
    "normalize_output_file_entry",
    "collect_expected_outputs",
    "collect_expected_columns",
    "match_expected_files",
    "stage_comparisons_for_stage",
    "analysis_reports_for_stage",
    "validate_analysis_reports",
    "breakdown_comparison_classifications",
    # metrics
    "log_agent_call",
    "record_discrepancy",
    # materials
    "materials_from_stage_outputs",
    "extract_materials_from_plan_assumptions",
    "deduplicate_materials",
    "load_material_database",
    "match_material_from_text",
    "format_validated_material",
    "extract_validated_materials",
    "format_material_checkpoint_question",
]




