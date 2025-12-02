"""
Agent Node Implementations - Stubs for LangGraph Workflow

Each function is a LangGraph node that:
1. Receives the current ReproState
2. Performs agent-specific processing (TODO: implement with LLM calls)
3. Updates relevant state fields
4. Returns the updated state

These are currently stubs showing expected state mutations.
See prompts/*.md for the corresponding agent system prompts.
"""

import csv
import json
import os
import re
import signal
from collections import defaultdict
from math import isfinite
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np

from schemas.state import (
    ReproState, 
    save_checkpoint, 
    check_context_before_node,
    initialize_progress_from_plan,
    sync_extracted_parameters,
    validate_state_for_node,
    get_plan_stage,
    get_progress_stage,
    DISCREPANCY_THRESHOLDS,
)

PROJECT_ROOT = Path(__file__).parent.parent
from src.prompts import build_agent_prompt
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════════
# Metrics Logging
# ═══════════════════════════════════════════════════════════════════════

def log_agent_call(agent_name: str, node_name: str, start_time: datetime):
    """
    Decorator to log agent calls to state['metrics'].
    
    Note: This is a simplified version. Ideally this would be a proper decorator,
    but for state-passing functions, we can just call a helper at the end.
    """
    def record_metric(state: ReproState, result_dict: Dict[str, Any] = None):
        if "metrics" not in state:
            state["metrics"] = {"agent_calls": [], "stage_metrics": []}
            
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        metric = {
            "agent": agent_name,
            "node": node_name,
            "stage_id": state.get("current_stage_id"),
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "verdict": result_dict.get("execution_verdict") or 
                       result_dict.get("physics_verdict") or 
                       result_dict.get("supervisor_verdict") or
                       result_dict.get("last_plan_review_verdict") or
                       None,
            "error": result_dict.get("run_error") if result_dict else None
        }
        
        if "agent_calls" not in state["metrics"]:
            state["metrics"]["agent_calls"] = []
            
        state["metrics"]["agent_calls"].append(metric)
        
    return record_metric


# ═══════════════════════════════════════════════════════════════════════
# Context Window Management
# ═══════════════════════════════════════════════════════════════════════

def _check_context_or_escalate(state: ReproState, node_name: str) -> Optional[Dict[str, Any]]:
    """
    Check context before LLM call. Returns state updates if escalation needed, None otherwise.
    
    This is called explicitly at the start of each agent node that makes LLM calls.
    If context is critical and cannot be auto-recovered, prepares escalation to user.
    
    Args:
        state: Current ReproState
        node_name: Name of the node about to make LLM call
        
    Returns:
        None if safe to proceed (or with minor auto-recovery applied)
        Dict with state updates if escalation to user is needed
    """
    check = check_context_before_node(state, node_name, auto_recover=True)
    
    if check["ok"]:
        # Safe to proceed, possibly with state updates from auto-recovery
        return check.get("state_updates") if check.get("state_updates") else None
    
    if check["escalate"]:
        # Must ask user - return state updates to trigger ask_user
        return {
            "pending_user_questions": [check["user_question"]],
            "awaiting_user_input": True,
            "ask_user_trigger": "context_overflow",
            "last_node_before_ask_user": node_name,
        }
    
    # Shouldn't reach here, but fallback to escalation
    return {
        "pending_user_questions": [f"Context overflow in {node_name}. How should we proceed?"],
        "awaiting_user_input": True,
        "ask_user_trigger": "context_overflow",
        "last_node_before_ask_user": node_name,
    }


def _validate_user_responses(trigger: str, responses: Dict[str, str], questions: List[str]) -> List[str]:
    """
    Validate user responses against expected format for the trigger type.
    
    Args:
        trigger: The ask_user_trigger value (e.g., "material_checkpoint")
        responses: Dict mapping question -> response
        questions: List of questions that were asked
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if not responses:
        errors.append("No responses provided")
        return errors
    
    # Get all response text (combined)
    all_responses = " ".join(str(r).upper() for r in responses.values())
    
    if trigger == "material_checkpoint":
        # Must contain one of: APPROVE, CHANGE_MATERIAL, CHANGE_DATABASE, NEED_HELP
        valid_keywords = ["APPROVE", "CHANGE_MATERIAL", "CHANGE_DATABASE", "NEED_HELP", "HELP", 
                         "YES", "NO", "REJECT", "CORRECT", "WRONG"]
        if not any(kw in all_responses for kw in valid_keywords):
            errors.append(
                "Response must contain one of: APPROVE, CHANGE_MATERIAL, CHANGE_DATABASE, or NEED_HELP"
            )
    
    elif trigger == "code_review_limit":
        valid_keywords = ["PROVIDE_HINT", "HINT", "SKIP", "STOP", "RETRY"]
        if not any(kw in all_responses for kw in valid_keywords):
            errors.append(
                "Response must contain one of: PROVIDE_HINT, SKIP_STAGE, or STOP"
            )
    
    elif trigger == "design_review_limit":
        valid_keywords = ["PROVIDE_HINT", "HINT", "SKIP", "STOP", "RETRY"]
        if not any(kw in all_responses for kw in valid_keywords):
            errors.append(
                "Response must contain one of: PROVIDE_HINT, SKIP_STAGE, or STOP"
            )
    
    elif trigger == "execution_failure_limit":
        valid_keywords = ["RETRY", "GUIDANCE", "SKIP", "STOP"]
        if not any(kw in all_responses for kw in valid_keywords):
            errors.append(
                "Response must contain one of: RETRY_WITH_GUIDANCE, SKIP_STAGE, or STOP"
            )
    
    elif trigger == "physics_failure_limit":
        valid_keywords = ["RETRY", "ACCEPT", "PARTIAL", "SKIP", "STOP"]
        if not any(kw in all_responses for kw in valid_keywords):
            errors.append(
                "Response must contain one of: RETRY_WITH_GUIDANCE, ACCEPT_PARTIAL, SKIP_STAGE, or STOP"
            )
    
    elif trigger == "backtrack_approval":
        valid_keywords = ["APPROVE", "REJECT", "YES", "NO"]
        if not any(kw in all_responses for kw in valid_keywords):
            errors.append(
                "Response must contain one of: APPROVE or REJECT"
            )
    
    elif trigger == "replan_limit":
        valid_keywords = ["FORCE", "ACCEPT", "GUIDANCE", "STOP"]
        if not any(kw in all_responses for kw in valid_keywords):
            errors.append(
                "Response must contain one of: FORCE_ACCEPT, PROVIDE_GUIDANCE, or STOP"
            )
    
    # For unknown triggers, just check that response is not empty
    elif trigger not in ["context_overflow", "backtrack_limit"]:
        if not all_responses.strip():
            errors.append("Response cannot be empty")
    
    return errors


def _validate_state_or_warn(state: ReproState, node_name: str) -> list:
    """
    Validate state for a node and return list of issues.
    
    This wraps validate_state_for_node() to provide consistent validation
    across agent nodes. Returns empty list if state is valid.
    
    Args:
        state: Current ReproState
        node_name: Name of the node about to execute
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = validate_state_for_node(state, node_name)
    if issues:
        import logging
        logger = logging.getLogger(__name__)
        for issue in issues:
            logger.warning(f"State validation issue for {node_name}: {issue}")
    return issues


def adapt_prompts_node(state: ReproState) -> ReproState:
    """PromptAdaptorAgent: Customize prompts for paper-specific needs."""
    context_update = _check_context_or_escalate(state, "adapt_prompts")
    if context_update:
        if context_update.get("awaiting_user_input"):
            return context_update  # type: ignore[return-value]
        state = {**state, **context_update}

    state["workflow_phase"] = "adapting_prompts"
    
    # Initialize empty adaptations list (to be populated by LLM)
    # This ensures the field exists for build_agent_prompt
    state["prompt_adaptations"] = []
    
    # TODO: Implement prompt adaptation logic
    # - Analyze paper domain and techniques
    # - Generate prompt modifications
    # - Store in state["prompt_adaptations"]
    return state


def _ensure_stub_figures(state: ReproState) -> List[Dict[str, Any]]:
    """Return available paper figures or a placeholder stub."""
    figures = state.get("paper_figures") or []
    if figures:
        return figures
    return [{
        "id": "FigStub",
        "description": "Placeholder figure generated for stub planning",
        "image_path": "",
    }]


def _build_stub_targets(figures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create target entries compatible with plan schema."""
    targets: List[Dict[str, Any]] = []
    for idx, fig in enumerate(figures):
        figure_id = fig.get("id") or f"Fig{idx + 1}"
        targets.append({
            "figure_id": figure_id,
            "description": fig.get("description", f"Simulation target {figure_id}"),
            "type": "spectrum",
            "simulation_class": "FDTD_DIRECT",
            "precision_requirement": "acceptable",
            "digitized_data_path": fig.get("digitized_data_path"),
        })
    if not targets:
        targets.append({
            "figure_id": "FigStub",
            "description": "Placeholder simulation target",
            "type": "spectrum",
            "simulation_class": "FDTD_DIRECT",
            "precision_requirement": "acceptable",
            "digitized_data_path": None,
        })
    return targets


def _build_stub_expected_outputs(paper_id: str, stage_id: str, target_ids: List[str], columns: List[str]) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for target in target_ids:
        outputs.append({
            "artifact_type": "spectrum_csv",
            "filename_pattern": f"{paper_id}_{stage_id}_{target.lower()}_spectrum.csv",
            "description": f"Simulation data for {target}",
            "columns": columns,
            "target_figure": target,
        })
    return outputs


def _build_stub_stages(paper_id: str, targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not targets:
        targets = [{
            "figure_id": "FigStub",
            "description": "Placeholder simulation target",
            "type": "spectrum",
            "simulation_class": "FDTD_DIRECT",
            "precision_requirement": "acceptable",
            "digitized_data_path": None,
        }]
    stage0_target = targets[0]["figure_id"]
    stage1_targets = [t["figure_id"] for t in targets[1:]] or [stage0_target]
    
    stage0 = {
        "stage_id": "stage0_material_validation",
        "stage_type": "MATERIAL_VALIDATION",
        "name": "Material optical properties validation",
        "description": "Validate material optical constants against primary reference figure.",
        "targets": [stage0_target],
        "dependencies": [],
        "is_mandatory_validation": True,
        "complexity_class": "analytical",
        "runtime_estimate_minutes": 2,
        "runtime_budget_minutes": 10,
        "max_revisions": 3,
        "fallback_strategy": "ask_user",
        "validation_criteria": [
            f"{stage0_target}: optical constants track reference within 10%"
        ],
        "expected_outputs": _build_stub_expected_outputs(
            paper_id,
            "stage0_material_validation",
            [stage0_target],
            ["wavelength_nm", "n", "k"]
        ),
        "reference_data_path": None,
    }
    
    stage1 = {
        "stage_id": "stage1_primary_structure",
        "stage_type": "SINGLE_STRUCTURE",
        "name": "Primary structure reproduction",
        "description": "Simulate the main structure described in the paper and compare spectra to referenced figures.",
        "targets": stage1_targets,
        "dependencies": ["stage0_material_validation"],
        "is_mandatory_validation": False,
        "complexity_class": "2D_light",
        "runtime_estimate_minutes": 15,
        "runtime_budget_minutes": 45,
        "max_revisions": 3,
        "fallback_strategy": "ask_user",
        "validation_criteria": [
            f"{target}: resonance within 5% of reference" for target in stage1_targets
        ],
        "expected_outputs": _build_stub_expected_outputs(
            paper_id,
            "stage1_primary_structure",
            stage1_targets,
            ["wavelength_nm", "transmission"]
        ),
        "reference_data_path": None,
    }
    
    return [stage0, stage1]


def _build_stub_planned_materials(state: ReproState) -> List[Dict[str, Any]]:
    domain = state.get("paper_domain", "generic")
    return [{
        "material_id": f"{domain}_placeholder",
        "name": f"{domain.title()} Material",
        "source": "stub",
        "path": "materials/placeholder.csv",
    }]


def _build_stub_assumptions() -> Dict[str, Any]:
    return {
        "global_assumptions": {
            "materials": [],
            "geometry": [],
            "sources": [],
        },
        "stage_specific": [],
    }


def _build_stub_plan(state: ReproState) -> Dict[str, Any]:
    paper_id = state.get("paper_id", "paper_stub")
    paper_title = state.get("paper_title", paper_id.replace("_", " ").title())
    figures = _ensure_stub_figures(state)
    targets = _build_stub_targets(figures)
    stages = _build_stub_stages(paper_id, targets)
    total_figures = len(figures)
    attempted = [t["figure_id"] for t in targets]
    coverage = 0.0
    if total_figures:
        coverage = round(len(attempted) / total_figures * 100, 2)
    
    plan = {
        "paper_id": paper_id,
        "paper_domain": state.get("paper_domain", "other"),
        "title": paper_title,
        "summary": f"Stub plan automatically generated for {paper_title}. Replace with PlannerAgent output.",
        "simulation_approach": "FDTD with Meep (stub)",
        "main_system": state.get("paper_domain", "other"),
        "targets": targets,
        "stages": stages,
        "reproduction_scope": {
            "total_figures": total_figures,
            "reproducible_figures": len(attempted),
            "reproducible_figure_ids": attempted,
            "attempted_figures": attempted,
            "skipped_figures": [],
            "coverage_percent": coverage,
        },
        "extracted_parameters": [
            {
                "name": "stub_dimension_nm",
                "value": 100,
                "unit": "nm",
                "source": "inferred",
                "location": "stub_generator",
                "cross_checked": False,
                "discrepancy_notes": "Placeholder parameter - replace with PlannerAgent output.",
            }
        ],
    }
    return plan


def _resolve_data_path(path_str: Optional[str]) -> Optional[Path]:
    """Resolve relative or absolute paths for analyzer inputs."""
    if not path_str:
        return None
    candidate = Path(path_str).expanduser()
    if candidate.exists():
        return candidate
    alt = PROJECT_ROOT / path_str
    if alt.exists():
        return alt
    return None


def _normalize_series(xs: List[float], ys: List[float]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not xs or not ys or len(xs) != len(ys):
        return None
    pairs = sorted(zip(xs, ys), key=lambda item: item[0])
    x_arr = np.array([float(p[0]) for p in pairs], dtype=float)
    y_arr = np.array([float(p[1]) for p in pairs], dtype=float)
    if len(x_arr) < 3:
        return None
    return x_arr, y_arr


def _units_to_multiplier(header: str) -> float:
    header_lower = header.lower()
    if header_lower.endswith("_nm") or "(nm)" in header_lower:
        return 1.0
    if header_lower.endswith("_um") or "(um)" in header_lower:
        return 1000.0
    if header_lower.endswith("_m") or "(m)" in header_lower:
        return 1e9
    return 1.0


def _load_numeric_series(path_str: Optional[str], columns: Optional[List[str]] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    resolved = _resolve_data_path(path_str)
    if not resolved or not resolved.exists():
        return None
    suffix = resolved.suffix.lower()
    try:
        if suffix in {".csv", ".tsv"}:
            delimiter = "," if suffix == ".csv" else "\t"
            xs: List[float] = []
            ys: List[float] = []
            with resolved.open("r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=delimiter)
                header = next(reader, None)
                x_idx = y_idx = None
                x_multiplier = y_multiplier = 1.0
                if header:
                    normalized_header = [h.strip() for h in header]
                    if columns and len(columns) >= 2:
                        col_x = columns[0]
                        col_y = columns[1]
                        for idx, field in enumerate(normalized_header):
                            field_lower = field.lower()
                            if col_x and col_x.lower() in field_lower and x_idx is None:
                                x_idx = idx
                                x_multiplier = _units_to_multiplier(field_lower)
                            if col_y and col_y.lower() in field_lower and y_idx is None:
                                y_idx = idx
                                y_multiplier = _units_to_multiplier(field_lower)
                    if x_idx is None:
                        x_idx = next((i for i, field in enumerate(normalized_header) if "wave" in field.lower()), None)
                        if x_idx is not None:
                            x_multiplier = _units_to_multiplier(normalized_header[x_idx])
                    if y_idx is None and len(normalized_header) > 1:
                        y_idx = 1 if x_idx == 0 else 0
                    if y_idx is None:
                        y_idx = x_idx + 1 if x_idx is not None and x_idx + 1 < len(normalized_header) else None
                for row in reader:
                    try:
                        if x_idx is not None and y_idx is not None and len(row) > y_idx:
                            x_val = float(row[x_idx]) * x_multiplier
                            y_val = float(row[y_idx]) * y_multiplier
                        else:
                            numeric_cells = [float(cell) for cell in row]
                            if len(numeric_cells) < 2:
                                continue
                            x_val = numeric_cells[0]
                            y_val = numeric_cells[1]
                        xs.append(x_val)
                        ys.append(y_val)
                    except (ValueError, IndexError):
                        continue
            return _normalize_series(xs, ys)
        if suffix == ".json":
            data = json.loads(resolved.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "x" in data and "y" in data:
                xs = [float(x) for x in data["x"]]
                ys = [float(y) for y in data["y"]]
                return _normalize_series(xs, ys)
            if isinstance(data, list):
                xs = []
                ys = []
                for point in data:
                    if isinstance(point, dict):
                        x_val = point.get("x")
                        y_val = point.get("y")
                    elif isinstance(point, (list, tuple)) and len(point) >= 2:
                        x_val, y_val = point[0], point[1]
                    else:
                        continue
                    try:
                        xs.append(float(x_val))
                        ys.append(float(y_val))
                    except (TypeError, ValueError):
                        continue
                return _normalize_series(xs, ys)
            return None
        if suffix == ".npy":
            arr = np.load(resolved)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return _normalize_series(arr[:, 0].tolist(), arr[:, 1].tolist())
            return None
        if suffix == ".npz":
            archive = np.load(resolved)
            first_key = archive.files[0] if archive.files else None
            if not first_key:
                return None
            arr = archive[first_key]
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return _normalize_series(arr[:, 0].tolist(), arr[:, 1].tolist())
            return None
    except Exception:
        return None
    return None


def _compute_peak_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute basic peak metrics for a curve."""
    if x.size == 0 or y.size == 0:
        return {}
    idx = int(np.argmax(y))
    peak_position = float(x[idx])
    peak_value = float(y[idx])
    half_max = peak_value / 2 if peak_value else None
    left_cross = None
    right_cross = None
    if half_max and half_max > 0:
        for i in range(idx, -1, -1):
            if y[i] <= half_max:
                left_cross = float(x[i])
                break
        for i in range(idx, len(y)):
            if y[i] <= half_max:
                right_cross = float(x[i])
                break
    fwhm = None
    if left_cross is not None and right_cross is not None:
        fwhm = abs(right_cross - left_cross)
    return {
        "peak_position": peak_position,
        "peak_value": peak_value,
        "fwhm": fwhm if fwhm is not None else None,
    }


def _quantitative_curve_metrics(
    sim_series: Optional[Tuple[np.ndarray, np.ndarray]],
    ref_series: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    """Compute quantitative metrics comparing simulation vs reference curves."""
    metrics: Dict[str, float] = {}
    if not sim_series:
        return metrics
    x_sim, y_sim = sim_series
    sim_peak = _compute_peak_metrics(x_sim, y_sim)
    metrics["peak_position_sim"] = sim_peak.get("peak_position")
    metrics["peak_value_sim"] = sim_peak.get("peak_value")
    metrics["fwhm_sim"] = sim_peak.get("fwhm")
    
    if not ref_series:
        return metrics
    
    x_ref, y_ref = ref_series
    if x_ref.size < 3 or y_ref.size < 3:
        return metrics
    ref_peak = _compute_peak_metrics(x_ref, y_ref)
    metrics["peak_position_paper"] = ref_peak.get("peak_position")
    metrics["peak_value_paper"] = ref_peak.get("peak_value")
    metrics["fwhm_paper"] = ref_peak.get("fwhm")
    
    # Interpolate reference onto simulation axis
    y_ref_interp = np.interp(x_sim, x_ref, y_ref, left=np.nan, right=np.nan)
    mask = np.isfinite(y_ref_interp)
    if not mask.any():
        return metrics
    sim_vals = y_sim[mask]
    ref_vals = y_ref_interp[mask]
    n_points = sim_vals.size
    metrics["n_points_compared"] = int(n_points)
    if n_points == 0:
        return metrics
    
    peak_paper = metrics.get("peak_position_paper")
    peak_sim = metrics.get("peak_position_sim")
    if peak_paper is not None and peak_sim is not None and peak_paper != 0:
        metrics["peak_position_error_percent"] = abs(peak_sim - peak_paper) / abs(peak_paper) * 100.0
    
    if metrics.get("peak_value_paper"):
        peak_ratio = metrics.get("peak_value_sim", 0) / metrics["peak_value_paper"]
        metrics["peak_height_ratio"] = float(peak_ratio) if peak_ratio and isfinite(peak_ratio) else None
    
    if metrics.get("fwhm_paper") and metrics.get("fwhm_sim"):
        paper_fwhm = metrics["fwhm_paper"]
        if paper_fwhm:
            metrics["fwhm_ratio"] = float(metrics["fwhm_sim"] / paper_fwhm)
    
    rmse = float(np.sqrt(np.mean((sim_vals - ref_vals) ** 2)))
    value_range = float(np.max(ref_vals) - np.min(ref_vals)) or 1.0
    metrics["normalized_rmse_percent"] = rmse / value_range * 100.0
    
    if n_points > 1:
        corr = np.corrcoef(sim_vals, ref_vals)[0, 1]
        if isfinite(corr):
            metrics["correlation"] = float(corr)
        mean_ref = float(np.mean(ref_vals))
        ss_res = float(np.sum((ref_vals - sim_vals) ** 2))
        ss_tot = float(np.sum((ref_vals - mean_ref) ** 2)) or 1.0
        r_squared = 1.0 - (ss_res / ss_tot)
        metrics["r_squared"] = float(r_squared)
    
    return metrics


def _classify_percent_error(error_percent: float) -> str:
    thresholds = DISCREPANCY_THRESHOLDS["resonance_wavelength"]
    if error_percent <= thresholds["excellent"]:
        return "match"
    if error_percent <= thresholds["acceptable"]:
        return "partial_match"
    return "mismatch"


def _classification_from_metrics(
    metrics: Dict[str, float],
    precision_requirement: str,
    has_reference: bool,
) -> str:
    if not has_reference:
        if precision_requirement == "excellent":
            return "pending_validation"
        # Without reference data, treat as qualitative-only pending further review
        return "pending_validation"
    if precision_requirement == "qualitative":
        return "match"
    
    error_percent = metrics.get("peak_position_error_percent")
    if error_percent is not None:
        return _classify_percent_error(error_percent)
    
    rmse = metrics.get("normalized_rmse_percent")
    if rmse is not None:
        if rmse <= 5:
            return "match"
        if rmse <= 15:
            return "partial_match"
        return "mismatch"
    
    return "pending_validation"


CRITERIA_PATTERNS = [
    ("resonance_within_percent", re.compile(r"resonance.*within\s+(\d+)%"), "peak_position_error_percent", "max"),
    ("peak_within_percent", re.compile(r"peak.*within\s+(\d+)%"), "peak_position_error_percent", "max"),
    ("normalized_rmse_max", re.compile(r"normalized\s*rmse.*(?:<=|less than)\s*(\d+)%"), "normalized_rmse_percent", "max"),
    ("correlation_min", re.compile(r"correlation.*(?:>=|greater than)\s*(0\.\d+|\d+(\.\d+)?)"), "correlation", "min"),
]


def _evaluate_validation_criteria(metrics: Dict[str, Any], criteria: List[str]) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    if not criteria:
        return True, failures
    for criterion in criteria:
        normalized = criterion.lower()
        matched_pattern = False
        for pattern_name, pattern, metric_key, mode in CRITERIA_PATTERNS:
            match = pattern.search(normalized)
            if not match:
                continue
            matched_pattern = True
            threshold = float(match.group(1))
            metric_value = metrics.get(metric_key)
            if metric_value is None:
                failures.append(f"{criterion.strip()} (missing metric '{metric_key}')")
            else:
                if mode == "max" and metric_value > threshold:
                    failures.append(f"{criterion.strip()} (measured {metric_value:.2f} > {threshold})")
                if mode == "min" and metric_value < threshold:
                    failures.append(f"{criterion.strip()} (measured {metric_value:.2f} < {threshold})")
            break
        if not matched_pattern:
            # Unknown format; treat as informational
            continue
    return (len(failures) == 0), failures


def _extract_targets_from_feedback(feedback: Optional[str], known_targets: List[str]) -> List[str]:
    if not feedback:
        return []
    pattern = re.compile(r"(Fig\s?[0-9]+[a-zA-Z]?)", re.IGNORECASE)
    matches = [m.strip().replace(" ", "") for m in pattern.findall(feedback)]
    ordered = []
    seen = set()
    normalized_known = {t.lower(): t for t in known_targets}
    for match in matches:
        normalized = match.lower()
        if normalized in normalized_known and normalized not in seen:
            ordered.append(normalized_known[normalized])
            seen.add(normalized)
    return ordered


def _match_output_file(file_entries: List[Any], target_id: str) -> Optional[str]:
    """Best-effort match simulation output files to a target figure."""
    normalized_id = (target_id or "").lower()
    for entry in file_entries:
        path_str = entry if isinstance(entry, str) else entry.get("path") or entry.get("file") or str(entry)
        name = Path(path_str).name.lower()
        if normalized_id and normalized_id in name:
            return path_str
    if file_entries:
        entry = file_entries[0]
        return entry if isinstance(entry, str) else entry.get("path") or entry.get("file")
    return None


def _normalize_output_file_entry(entry: Any) -> Optional[str]:
    """Return a string path for a stage output entry."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("path") or entry.get("file") or entry.get("filename")
    return str(entry) if entry is not None else None


def _collect_expected_outputs(plan_stage: Optional[Dict[str, Any]], paper_id: str, stage_id: Optional[str]) -> Dict[str, List[str]]:
    """Build a mapping of target_id -> list of expected filenames for the stage."""
    mapping: Dict[str, List[str]] = defaultdict(list)
    if not plan_stage:
        return mapping
    expected = plan_stage.get("expected_outputs") or []
    for spec in expected:
        target = spec.get("target_figure")
        pattern = spec.get("filename_pattern")
        if not target or not pattern:
            continue
        resolved = pattern.replace("{paper_id}", paper_id or "")
        if stage_id:
            resolved = resolved.replace("{stage_id}", stage_id)
        resolved = resolved.replace("{target_id}", target.lower())
        mapping[target].append(resolved)
    return mapping


def _collect_expected_columns(plan_stage: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Map target_id -> ordered column names, when provided."""
    mapping: Dict[str, List[str]] = {}
    if not plan_stage:
        return mapping
    expected = plan_stage.get("expected_outputs") or []
    for spec in expected:
        target = spec.get("target_figure")
        columns = spec.get("columns")
        if target and columns:
            mapping[target] = columns
    return mapping


def _match_expected_files(expected_filenames: List[str], output_files: List[Any]) -> Optional[str]:
    """Match actual files against the plan's expected filenames."""
    if not expected_filenames:
        return None
    normalized_outputs = [
        Path(_normalize_output_file_entry(entry) or "").name
        for entry in output_files
    ]
    path_lookup = {
        Path(_normalize_output_file_entry(entry) or "").name: _normalize_output_file_entry(entry)
        for entry in output_files
    }
    for expected in expected_filenames:
        expected_name = Path(expected).name
        if expected_name in path_lookup:
            return path_lookup[expected_name]
    # Allow substring match if exact name missing
    for expected in expected_filenames:
        expected_name = Path(expected).name
        for actual in normalized_outputs:
            if expected_name in actual:
                return path_lookup.get(actual)
    return None


def _record_discrepancy(
    state: ReproState,
    stage_id: Optional[str],
    figure_id: str,
    quantity: str,
    paper_value: str,
    simulation_value: str,
    classification: str = "investigate",
    difference_percent: float = 100.0,
    likely_cause: str = "",
    action_taken: str = "",
    blocking: bool = True,
) -> Dict[str, Any]:
    """Append a discrepancy entry to both global log and the stage progress."""
    if state.get("discrepancies_log") is None:
        state["discrepancies_log"] = []
    log = state.setdefault("discrepancies_log", [])
    entry_id = f"D{len(log) + 1}"
    discrepancy = {
        "id": entry_id,
        "figure": figure_id,
        "quantity": quantity,
        "paper_value": paper_value,
        "simulation_value": simulation_value,
        "difference_percent": difference_percent,
        "classification": classification,
        "likely_cause": likely_cause,
        "action_taken": action_taken,
        "blocking": blocking,
    }
    log.append(discrepancy)
    
    if stage_id:
        progress_stage = get_progress_stage(state, stage_id)
        if progress_stage is not None:
            stage_discrepancies = progress_stage.setdefault("discrepancies", [])
            stage_discrepancies.append(discrepancy)
    return discrepancy


def _materials_from_stage_outputs(state: ReproState) -> List[Dict[str, Any]]:
    """Build material entries from Stage 0 artifacts (live or archived)."""
    def _build_from_entries(entries: List[Any], source_label: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for entry in entries:
            path_str = _normalize_output_file_entry(entry)
            if not path_str:
                continue
            suffix = Path(path_str).suffix.lower()
            if suffix not in {".csv", ".json", ".h5", ".hdf5", ".npz", ".npy"}:
                continue
            material_id = Path(path_str).stem
            results.append({
                "material_id": material_id,
                "name": material_id.replace("_", " ").title(),
                "source": source_label,
                "path": path_str,
                "csv_available": suffix == ".csv",
                "from": source_label,
            })
        return results
    
    stage_outputs = state.get("stage_outputs", {})
    files = stage_outputs.get("files", [])
    materials = _build_from_entries(files, "stage0_output")
    
    if not materials:
        stage0_progress = get_progress_stage(state, "stage0_material_validation")
        progress_files = []
        if stage0_progress:
            for output_entry in stage0_progress.get("outputs", []):
                filename = output_entry.get("filename")
                if filename:
                    progress_files.append(filename)
        materials = _build_from_entries(progress_files, "stage0_progress")
    
    return materials


def _extract_materials_from_plan_assumptions(state: ReproState) -> List[Dict[str, Any]]:
    """Existing fallback path that scans plan parameters and assumptions."""
    import json
    import os
    
    plan = state.get("plan", {})
    extracted_params = plan.get("extracted_parameters", [])
    assumptions = state.get("assumptions", {})
    
    # Load material database
    material_db = _load_material_database()
    if not material_db:
        return []
    
    material_lookup = {}
    for mat in material_db.get("materials", []):
        mat_id = mat.get("material_id", "")
        material_lookup[mat_id] = mat
        parts = mat_id.split("_")
        if len(parts) >= 2:
            simple_name = parts[-1]
            if simple_name not in material_lookup:
                material_lookup[simple_name] = mat
    
    validated_materials = []
    seen_material_ids = set()
    
    for param in extracted_params:
        name = param.get("name", "").lower()
        value = str(param.get("value", "")).lower()
        if "material" in name:
            matched_material = _match_material_from_text(value, material_lookup)
            if matched_material and matched_material["material_id"] not in seen_material_ids:
                validated_materials.append(_format_validated_material(
                    matched_material, 
                    from_source=f"parameter: {param.get('name')}"
                ))
                seen_material_ids.add(matched_material["material_id"])
    
    global_assumptions = assumptions.get("global_assumptions", {})
    material_assumptions = global_assumptions.get("materials", [])
    
    for assumption in material_assumptions:
        if isinstance(assumption, dict):
            desc = assumption.get("description", "").lower()
            matched_material = _match_material_from_text(desc, material_lookup)
            if matched_material and matched_material["material_id"] not in seen_material_ids:
                validated_materials.append(_format_validated_material(
                    matched_material,
                    from_source=f"assumption: {assumption.get('description', '')[:50]}"
                ))
                seen_material_ids.add(matched_material["material_id"])
    
    return validated_materials


def _deduplicate_materials(materials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for mat in materials:
        key = mat.get("path") or mat.get("material_id")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(mat)
    return deduped


def _stage_comparisons_for_stage(state: ReproState, stage_id: Optional[str]) -> List[Dict[str, Any]]:
    """Return figure comparison entries associated with a specific stage."""
    if not stage_id:
        return []
    return [
        comp for comp in state.get("figure_comparisons", [])
        if comp.get("stage_id") == stage_id
    ]


def _analysis_reports_for_stage(state: ReproState, stage_id: Optional[str]) -> List[Dict[str, Any]]:
    if not stage_id:
        return []
    return [
        report for report in state.get("analysis_result_reports", [])
        if report.get("stage_id") == stage_id
    ]


def _validate_analysis_reports(reports: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    for report in reports:
        status = (report.get("status") or "").lower()
        metrics = report.get("quantitative_metrics") or {}
        precision = (report.get("precision_requirement") or "").lower()
        target = report.get("target_figure", "unknown")
        error = metrics.get("peak_position_error_percent")
        if precision == "excellent" and not metrics:
            issues.append(f"{target}: excellent precision requires quantitative metrics.")
            continue
        if error is not None:
            thresholds = DISCREPANCY_THRESHOLDS["resonance_wavelength"]
            if status == "match" and error > thresholds["acceptable"]:
                issues.append(f"{target}: classified as match but peak error {error:.2f}% > acceptable {thresholds['acceptable']}%.")
            if status in {"pending_validation", "partial_match"} and error > thresholds["investigate"]:
                issues.append(f"{target}: error {error:.2f}% exceeds investigate threshold; classification should be 'mismatch'.")
        failures = report.get("criteria_failures") or []
        for failure in failures:
            issues.append(f"{target}: {failure}")
    return issues


def _breakdown_comparison_classifications(comparisons: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Summarize comparison classifications into missing/pending/match buckets."""
    breakdown: Dict[str, List[str]] = {"missing": [], "pending": [], "matches": []}
    for comp in comparisons:
        classification = (comp.get("classification") or "").lower()
        figure_id = comp.get("figure_id") or "unknown"
        if classification in {"missing_output", "fail", "not_reproduced", "mismatch", "poor_match"}:
            breakdown["missing"].append(figure_id)
        elif classification in {"pending_validation", "partial_match", "match_pending", "partial"}:
            breakdown["pending"].append(figure_id)
        else:
            breakdown["matches"].append(figure_id)
    return breakdown


def plan_node(state: ReproState) -> dict:
    """
    PlannerAgent: Analyze paper and create reproduction plan.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: This node makes LLM calls with full paper text, so it must 
    check context first. The planner receives the largest context of any node.
    """
    start_time = datetime.now(timezone.utc)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT CHECK: CRITICAL for planner - receives full paper text
    # ═══════════════════════════════════════════════════════════════════════
    escalation = _check_context_or_escalate(state, "plan")
    if escalation is not None:
        # Context overflow - return escalation state updates
        return escalation
    
    # TODO: Implement planning logic
    # - Extract parameters from paper
    # - Classify figures
    # - Design staged reproduction plan
    # - Initialize assumptions
    # - Call LLM with planner_agent.md prompt
    # - Parse agent output per planner_output_schema.json
    
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("planner", state)
    
    # Inject replan context for learning
    replan_count = state.get("replan_count", 0)
    if replan_count > 0:
        system_prompt += f"\n\nNOTE: This is Replan Attempt #{replan_count}. Previous plan was rejected. Improve strategy based on feedback."
    
    # STUB: Replace with actual LLM call that populates plan, assumptions, etc.
    plan_data = _build_stub_plan(state)
    planned_materials = state.get("planned_materials") or _build_stub_planned_materials(state)
    assumptions = state.get("assumptions") or _build_stub_assumptions()
    
    result = {
        "workflow_phase": "planning",
        "plan": plan_data,
        "planned_materials": planned_materials,
        "assumptions": assumptions,
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # MANDATORY: Initialize progress stages from plan (after plan is set)
    # This converts plan stages into progress stages with status tracking.
    # Must be called before select_stage_node runs.
    # ═══════════════════════════════════════════════════════════════════════
    if result.get("plan") and result["plan"].get("stages"):
        state_with_plan = {**state, **result}
        # Force reset of progress on replan to ensure sync
        if "progress" in state_with_plan:
            # Backup old progress for history if needed, but clear active progress
            state_with_plan["progress"] = None 
            
        state_with_plan = initialize_progress_from_plan(state_with_plan)
        state_with_plan = sync_extracted_parameters(state_with_plan)
        result["progress"] = state_with_plan.get("progress")
        result["extracted_parameters"] = state_with_plan.get("extracted_parameters")
    
    # Log metrics
    log_agent_call("PlannerAgent", "plan", start_time)(state, result)
    
    return result


def plan_reviewer_node(state: ReproState) -> dict:
    """
    PlanReviewerAgent: Review reproduction plan before stage execution.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT:
    - Sets `last_plan_review_verdict` state field.
    - Increments `replan_count` when verdict is "needs_revision".
    The routing function `route_after_plan_review` reads these fields.
    - Validates plan precision requirements (excellent targets need digitized data).
    """
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT CHECK: Plan review receives large artifacts; guard context window
    # ═══════════════════════════════════════════════════════════════════════
    context_update = _check_context_or_escalate(state, "plan_review")
    if context_update:
        if context_update.get("awaiting_user_input"):
            return context_update
        state = {**state, **context_update}

    # ═══════════════════════════════════════════════════════════════════════
    # STATE VALIDATION: Check plan meets requirements before review
    # ═══════════════════════════════════════════════════════════════════════
    validation_issues = _validate_state_or_warn(state, "plan_review")
    
    # Separate blocking issues (PLAN_ISSUE) from other validation warnings
    blocking_issues = [i for i in validation_issues if i.startswith("PLAN_ISSUE:")]
    
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("plan_reviewer", state)
    
    # TODO: Implement plan review logic using prompts/plan_reviewer_agent.md
    # - Check coverage of reproducible figures
    # - Verify Stage 0 and Stage 1 present
    # - Validate parameter extraction
    # - Check assumptions quality
    # - Verify runtime estimates
    # - Call LLM with plan_reviewer_agent.md prompt
    # - Parse agent output per plan_reviewer_output_schema.json
    
    # ═══════════════════════════════════════════════════════════════════════
    # VALIDATE PLAN HAS STAGES: Critical check before approval
    # ═══════════════════════════════════════════════════════════════════════
    plan = state.get("plan", {})
    plan_stages = plan.get("stages", [])
    
    if not plan_stages or len(plan_stages) == 0:
        # Plan has no stages - this is a blocking issue
        blocking_issues.append(
            "PLAN_ISSUE: Plan must contain at least one stage. "
            "Current plan has no stages defined."
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # DETECT CIRCULAR DEPENDENCIES: Prevent infinite loops
    # ═══════════════════════════════════════════════════════════════════════
    if plan_stages:
        stage_ids = {s.get("stage_id") for s in plan_stages if s.get("stage_id")}
        
        def detect_cycles() -> List[List[str]]:
            """Detect circular dependencies using DFS."""
            cycles = []
            visited = set()
            rec_stack = set()
            path = []
            
            def dfs(stage_id: str):
                if stage_id in rec_stack:
                    # Found a cycle - extract the cycle path
                    cycle_start = path.index(stage_id)
                    cycle = path[cycle_start:] + [stage_id]
                    cycles.append(cycle)
                    return
                
                if stage_id in visited:
                    return
                
                visited.add(stage_id)
                rec_stack.add(stage_id)
                path.append(stage_id)
                
                # Find stage and check its dependencies
                stage = next((s for s in plan_stages if s.get("stage_id") == stage_id), None)
                if stage:
                    dependencies = stage.get("dependencies", [])
                    for dep_id in dependencies:
                        if dep_id in stage_ids:  # Only check if dependency exists
                            dfs(dep_id)
                
                path.pop()
                rec_stack.remove(stage_id)
            
            # Check all stages
            for stage_id in stage_ids:
                if stage_id not in visited:
                    dfs(stage_id)
            
            return cycles
        
        cycles = detect_cycles()
        if cycles:
            cycle_descriptions = [
                " → ".join(cycle) + " (circular)"
                for cycle in cycles
            ]
            blocking_issues.append(
                f"PLAN_ISSUE: Circular dependencies detected: {', '.join(cycle_descriptions)}. "
                "Stages cannot depend on themselves or form dependency cycles. "
                "Please fix the dependency graph."
            )
        
        # Also check for self-dependencies
        for stage in plan_stages:
            stage_id = stage.get("stage_id")
            dependencies = stage.get("dependencies", [])
            if stage_id in dependencies:
                blocking_issues.append(
                    f"PLAN_ISSUE: Stage '{stage_id}' depends on itself. "
                    "Stages cannot depend on themselves."
                )
    
    # If there are blocking plan issues (e.g., excellent precision without digitized data),
    # automatically flag for revision
    if blocking_issues:
        agent_output = {
            "verdict": "needs_revision",
            "issues": [{"severity": "blocking", "description": issue} for issue in blocking_issues],
            "summary": f"Plan has {len(blocking_issues)} blocking issue(s) requiring revision",
            "feedback": "The following issues must be resolved:\n" + "\n".join(blocking_issues),
        }
    else:
        # STUB: Replace with actual LLM call
        agent_output = {
            "verdict": "approve",  # "approve" | "needs_revision"
            "issues": [],
            "summary": "Plan review stub - implement with LLM call",
        }
    
    result = {
        "workflow_phase": "plan_review",
        "last_plan_review_verdict": agent_output["verdict"],
    }
    
    # Increment replan counter if needs_revision
    if agent_output["verdict"] == "needs_revision":
        result["replan_count"] = state.get("replan_count", 0) + 1
        result["planner_feedback"] = agent_output.get("feedback", agent_output.get("summary", ""))
    
    return result


def select_stage_node(state: ReproState) -> dict:
    """
    Select next stage based on dependencies and validation hierarchy.
    
    Priority order:
    1. Stages with status "needs_rerun" (highest priority - backtrack targets)
    2. Stages with status "not_started" whose dependencies are satisfied
    3. No more stages to run (return None for current_stage_id)
    
    Skips stages with status:
    - "completed_success" / "completed_partial" / "completed_failed" - already done
    - "invalidated" - waiting for dependency to complete
    - "in_progress" - shouldn't happen, but skip
    - "blocked" - dependencies not met or budget exceeded
    
    Returns:
        Dict with state updates (LangGraph merges this into state)
    """
    from schemas.state import get_validation_hierarchy, STAGE_TYPE_TO_HIERARCHY_KEY
    
    progress = state.get("progress", {})
    stages = progress.get("stages", [])
    plan = state.get("plan", {})
    plan_stages = plan.get("stages", [])
    
    if not stages and not plan_stages:
        # No stages defined yet - this is an error state
        # Should not happen if plan_review validated correctly, but handle gracefully
        import logging
        logger = logging.getLogger(__name__)
        logger.error(
            "select_stage_node called but no stages exist in plan or progress. "
            "This indicates plan_review failed to validate or plan initialization failed."
        )
        return {
            "workflow_phase": "stage_selection",
            "current_stage_id": None,
            "current_stage_type": None,
            "ask_user_trigger": "no_stages_available",
            "pending_user_questions": [
                "ERROR: No stages available to execute. The plan appears to be empty. "
                "Please check the plan and replan if necessary."
            ],
            "awaiting_user_input": True,
        }
    
    # Use plan stages if progress stages aren't initialized
    if not stages:
        stages = plan_stages
    
    # Additional validation: ensure stages list is not empty after initialization
    if not stages or len(stages) == 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.error("Stages list is empty after initialization - this should not happen")
        return {
            "workflow_phase": "stage_selection",
            "current_stage_id": None,
            "current_stage_type": None,
            "ask_user_trigger": "no_stages_available",
            "pending_user_questions": [
                "ERROR: Stages list is empty. Cannot proceed with reproduction."
            ],
            "awaiting_user_input": True,
        }
    
    # Get current validation hierarchy
    hierarchy = get_validation_hierarchy(state)
    
    # Priority 1: Find stages that need re-run (backtrack targets)
    # CRITICAL: Check that dependencies are not invalidated before selecting needs_rerun stage
    for stage in stages:
        if stage.get("status") == "needs_rerun":
            # Guard against race condition: ensure no dependencies are invalidated
            dependencies = stage.get("dependencies", [])
            has_invalidated_deps = False
            for dep_id in dependencies:
                dep_stage = next((s for s in stages if s.get("stage_id") == dep_id), None)
                if dep_stage and dep_stage.get("status") == "invalidated":
                    has_invalidated_deps = True
                    break
            
            if has_invalidated_deps:
                # Dependencies are invalidated - wait for them to be re-run first
                # This stage will be selected once dependencies complete
                continue
            
            return {
                "workflow_phase": "stage_selection",
                "current_stage_id": stage.get("stage_id"),
                "current_stage_type": stage.get("stage_type"),
                # Reset per-stage counters when entering a new stage
                "design_revision_count": 0,
                "code_revision_count": 0,
                "execution_failure_count": 0,
                "physics_failure_count": 0,
                "analysis_revision_count": 0,
                # Reset per-stage outputs (prevent stale data from previous stage)
                "stage_outputs": {},
                # CRITICAL: Clear previous run_error to prevent stale failure signals
                "run_error": None,
                "analysis_summary": None,
                "analysis_overall_classification": None,
            }
    
    # Priority 2: Find not_started stages with satisfied dependencies
    for stage in stages:
        status = stage.get("status", "not_started")
        
        # Skip completed, in_progress, or blocked stages.
        # NOTE: "invalidated" stages ARE eligible if their dependencies are met (re-run)
        if status in ["completed_success", "completed_partial", "completed_failed", 
                      "in_progress", "blocked"]:
            continue
        
        # Check if dependencies are satisfied
        dependencies = stage.get("dependencies", [])
        deps_satisfied = True
        missing_deps = []
        
        # Build stage_id lookup for validation
        stage_ids = {s.get("stage_id") for s in stages if s.get("stage_id")}
        
        for dep_id in dependencies:
            # ═══════════════════════════════════════════════════════════════════════
            # VALIDATE DEPENDENCY EXISTS: Check dependency stage_id exists in plan
            # ═══════════════════════════════════════════════════════════════════════
            if dep_id not in stage_ids:
                missing_deps.append(dep_id)
                deps_satisfied = False
                continue
            
            # Find the dependency stage
            dep_stage = next((s for s in stages if s.get("stage_id") == dep_id), None)
            if dep_stage:
                dep_status = dep_stage.get("status", "not_started")
                # Dependency must be completed (success or partial)
                if dep_status not in ["completed_success", "completed_partial"]:
                    deps_satisfied = False
                    break
            else:
                # Stage ID exists but stage not found - this is an error
                missing_deps.append(dep_id)
                deps_satisfied = False
        
        # If dependencies are missing, mark stage as blocked
        if missing_deps:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"Stage '{stage.get('stage_id')}' has missing dependencies: {missing_deps}. "
                "This indicates a plan inconsistency. Marking stage as blocked."
            )
            # Update progress to mark stage as blocked
            from schemas.state import get_progress_stage, update_progress_stage_status
            progress_stage = get_progress_stage(state, stage.get("stage_id"))
            if progress_stage and progress_stage.get("status") != "blocked":
                update_progress_stage_status(
                    state,
                    stage.get("stage_id"),
                    "blocked",
                    summary=f"Blocked: Missing dependencies {missing_deps}"
                )
            continue
        
        if not deps_satisfied:
            continue
        
        # Check validation hierarchy for stage type
        stage_type = stage.get("stage_type")
        
        # Use consistent hierarchy keys from state.py
        # Note: get_validation_hierarchy returns keys like 'stage0_material_validation' (specific)
        # or abstract keys. The current implementation of get_validation_hierarchy in state.py
        # returns a dict with keys corresponding to the hierarchy levels.
        # We need to re-fetch hierarchy here as it might change if we processed multiple stages.
        hierarchy = get_validation_hierarchy(state)
        
        # Use schema-defined keys for robustness
        MAT_VAL_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["MATERIAL_VALIDATION"]
        SINGLE_STRUCT_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["SINGLE_STRUCTURE"]
        ARRAY_SYS_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["ARRAY_SYSTEM"]
        PARAM_SWEEP_KEY = STAGE_TYPE_TO_HIERARCHY_KEY["PARAMETER_SWEEP"]
        
        # Enforce hierarchy using STAGE_TYPE_TO_HIERARCHY_KEY mapping
        # This ensures robustness against schema changes
        required_level_key = STAGE_TYPE_TO_HIERARCHY_KEY.get(stage_type)
        
        if required_level_key:
            # Map current stage type to its prerequisite level in the hierarchy
            # e.g., SINGLE_STRUCTURE needs 'material_validation' to be passed
            if stage_type == "SINGLE_STRUCTURE":
                if hierarchy.get(MAT_VAL_KEY) not in ["passed", "partial"]:
                    continue
            elif stage_type == "ARRAY_SYSTEM":
                if hierarchy.get(SINGLE_STRUCT_KEY) not in ["passed", "partial"]:
                    continue
            elif stage_type == "PARAMETER_SWEEP":
                # Parameter sweeps typically need at least single structure
                if hierarchy.get(SINGLE_STRUCT_KEY) not in ["passed", "partial"]:
                    continue
            elif stage_type == "COMPLEX_PHYSICS":
                 if hierarchy.get(PARAM_SWEEP_KEY) not in ["passed", "partial"] and \
                    hierarchy.get(ARRAY_SYS_KEY) not in ["passed", "partial"]:
                    continue
        
        # This stage is eligible
        return {
            "workflow_phase": "stage_selection",
            "current_stage_id": stage.get("stage_id"),
            "current_stage_type": stage_type,
            "stage_start_time": datetime.now(timezone.utc).isoformat(),
            # Reset per-stage counters when entering a new stage
            "design_revision_count": 0,
            "code_revision_count": 0,
            "execution_failure_count": 0,
            "physics_failure_count": 0,
            "analysis_revision_count": 0,
            # Reset per-stage outputs (prevent stale data from previous stage)
            "stage_outputs": {},
            # CRITICAL: Clear previous run_error to prevent stale failure signals
            "run_error": None,
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # DEADLOCK DETECTION: Check if all remaining stages are permanently blocked
    # ═══════════════════════════════════════════════════════════════════════
    # Before returning None, check if this is a deadlock vs normal completion
    remaining_stages = [
        s for s in stages
        if s.get("status") not in ["completed_success", "completed_partial"]
    ]
    
    if remaining_stages:
        # Check if any remaining stages could potentially run
        potentially_runnable = []
        permanently_blocked = []
        
        for stage in remaining_stages:
            status = stage.get("status", "not_started")
            if status in ["not_started", "invalidated", "needs_rerun"]:
                potentially_runnable.append(stage.get("stage_id"))
            elif status in ["blocked", "completed_failed"]:
                permanently_blocked.append(stage.get("stage_id"))
        
        if not potentially_runnable and permanently_blocked:
            # All remaining stages are permanently blocked - this is a deadlock
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Deadlock detected: All remaining stages are permanently blocked or failed. "
                f"Blocked stages: {permanently_blocked}. Cannot proceed with reproduction."
            )
            return {
                "workflow_phase": "stage_selection",
                "current_stage_id": None,
                "current_stage_type": None,
                "ask_user_trigger": "deadlock_detected",
                "pending_user_questions": [
                    f"Deadlock detected: All remaining stages ({len(permanently_blocked)}) are permanently blocked or failed. "
                    f"Blocked stages: {', '.join(permanently_blocked[:5])}{'...' if len(permanently_blocked) > 5 else ''}. "
                    "Options: 1) Generate report with current results, 2) Replan to fix blocked stages, 3) Stop."
                ],
                "awaiting_user_input": True,
            }
    
    # No more stages to run - normal completion
    return {
        "workflow_phase": "stage_selection",
        "current_stage_id": None,
        "current_stage_type": None,
    }


def simulation_designer_node(state: ReproState) -> dict:
    """
    SimulationDesignerAgent: Design simulation setup for current stage.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: This node makes LLM calls, so it must check context first.
    """
    start_time = datetime.now(timezone.utc)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT CHECK: Required for all nodes that make LLM calls
    # ═══════════════════════════════════════════════════════════════════════
    escalation = _check_context_or_escalate(state, "design")
    if escalation is not None:
        # Context overflow - return escalation state updates
        return escalation
    
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("simulation_designer", state)
    
    # Inject complexity class from plan
    from schemas.state import get_stage_design_spec
    complexity_class = get_stage_design_spec(state, state.get("current_stage_id"), "complexity_class", "unknown")
    
    # TODO: Implement design logic
    # - Interpret geometry from plan
    # - Use complexity_class ({complexity_class}) to determine resolution/dimensions
    # - Select materials from validated_materials (Stage 1+) or planned_materials (Stage 0)
    # - Configure sources, BCs, monitors
    # - Estimate performance
    # - Call LLM with simulation_designer_agent.md prompt
    # - Parse agent output per simulation_designer_output_schema.json
    
    # STUB: Replace with actual LLM call
    result = {
        "workflow_phase": "design",
        "design_description": "STUB: Design description would be generated here."
        # agent_output fields would go here
    }
    
    log_agent_call("SimulationDesignerAgent", "design", start_time)(state, result)
    return result


def design_reviewer_node(state: ReproState) -> dict:
    """
    DesignReviewerAgent: Review simulation design before code generation.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT:
    - Sets `last_design_review_verdict` state field.
    - Increments `design_revision_count` when verdict is "needs_revision".
    The routing function `route_after_design_review` reads these fields.
    """
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT CHECK: Reviews can accumulate long histories; guard window
    # ═══════════════════════════════════════════════════════════════════════
    context_update = _check_context_or_escalate(state, "design_review")
    if context_update:
        if context_update.get("awaiting_user_input"):
            return context_update
        state = {**state, **context_update}

    # Connect prompt adaptation
    system_prompt = build_agent_prompt("design_reviewer", state)
    
    # TODO: Implement design review logic using prompts/design_reviewer_agent.md
    # - Check geometry matches paper
    # - Verify physics setup is correct
    # - Validate material choices
    # - Check unit system (a_unit)
    # - Verify source/excitation setup
    # - Call LLM with design_reviewer_agent.md prompt
    # - Parse agent output per design_reviewer_output_schema.json
    
    # STUB: Replace with actual LLM call
    agent_output = {
        "verdict": "approve",  # "approve" | "needs_revision"
        "issues": [],
        "summary": "Design review stub - implement with LLM call",
    }
    
    result = {
        "workflow_phase": "design_review",
        "last_design_review_verdict": agent_output["verdict"],
        "reviewer_issues": agent_output.get("issues", []),
    }
    
    # Increment design revision counter if needs_revision
    if agent_output["verdict"] == "needs_revision":
        result["design_revision_count"] = state.get("design_revision_count", 0) + 1
        result["reviewer_feedback"] = agent_output.get("feedback", agent_output.get("summary", ""))
    
    return result


def code_reviewer_node(state: ReproState) -> dict:
    """
    CodeReviewerAgent: Review generated code before execution.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT:
    - Sets `last_code_review_verdict` state field.
    - Increments `code_revision_count` when verdict is "needs_revision".
    The routing function `route_after_code_review` reads these fields.
    """
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT CHECK: Code + design can exceed window after revisions
    # ═══════════════════════════════════════════════════════════════════════
    context_update = _check_context_or_escalate(state, "code_review")
    if context_update:
        if context_update.get("awaiting_user_input"):
            return context_update
        state = {**state, **context_update}

    # Connect prompt adaptation
    system_prompt = build_agent_prompt("code_reviewer", state)
    
    # TODO: Implement code review logic using prompts/code_reviewer_agent.md
    # - Verify a_unit matches design
    # - Check Meep API usage
    # - Validate numerics implementation
    # - Check code quality (no plt.show, etc.)
    # - Call LLM with code_reviewer_agent.md prompt
    # - Parse agent output per code_reviewer_output_schema.json
    
    # STUB: Replace with actual LLM call
    agent_output = {
        "verdict": "approve",  # "approve" | "needs_revision"
        "issues": [],
        "summary": "Code review stub - implement with LLM call",
    }
    
    result = {
        "workflow_phase": "code_review",
        "last_code_review_verdict": agent_output["verdict"],
        "reviewer_issues": agent_output.get("issues", []),
    }
    
    # Increment code revision counter if needs_revision
    if agent_output["verdict"] == "needs_revision":
        result["code_revision_count"] = state.get("code_revision_count", 0) + 1
        result["reviewer_feedback"] = agent_output.get("feedback", agent_output.get("summary", ""))
    
    return result


def code_generator_node(state: ReproState) -> dict:
    """
    CodeGeneratorAgent: Generate Python+Meep code from approved design.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: This node makes LLM calls, so it must check context first.
    
    For Stage 1+, code generator MUST read material paths from 
    state["validated_materials"], NOT hardcode paths. validated_materials 
    is populated after Stage 0 + user approval.
    """
    start_time = datetime.now(timezone.utc)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT CHECK: Required for all nodes that make LLM calls
    # ═══════════════════════════════════════════════════════════════════════
    escalation = _check_context_or_escalate(state, "generate_code")
    if escalation is not None:
        # Context overflow - return escalation state updates
        return escalation
    
    # Connect prompt adaptation
    system_prompt = build_agent_prompt("code_generator", state)
    
    # ═══════════════════════════════════════════════════════════════════════
    # VALIDATE MATERIALS FOR STAGE 1+: Ensure validated_materials exists
    # ═══════════════════════════════════════════════════════════════════════
    current_stage_type = state.get("current_stage_type", "")
    if current_stage_type != "MATERIAL_VALIDATION":
        # Stage 1+ requires validated materials
        validated_materials = state.get("validated_materials", [])
        if not validated_materials:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"validated_materials is empty for Stage 1+ ({current_stage_type}). "
                "Code generation cannot proceed without validated materials."
            )
            return {
                "workflow_phase": "code_generation",
                "run_error": (
                    f"validated_materials is empty but required for {current_stage_type} code generation. "
                    "This indicates material_checkpoint_node did not run or user did not approve materials. "
                    "Check that Stage 0 completed and user confirmation was received."
                ),
                "code_revision_count": state.get("code_revision_count", 0) + 1,
            }
    
    # TODO: Implement code generation logic
    # - Convert design to Meep code
    # - For Stage 1+: Read material paths from state["validated_materials"]
    # - Include progress prints
    # - Set expected outputs per stage specification
    # - Call LLM with code_generator_agent.md prompt
    # - Parse agent output per code_generator_output_schema.json
    
    # STUB: Replace with actual LLM call
    generated_code = "# STUB: Simulation code would be generated here."
    
    # ═══════════════════════════════════════════════════════════════════════
    # VALIDATE GENERATED CODE: Ensure code is non-empty and not stub
    # ═══════════════════════════════════════════════════════════════════════
    stub_markers = ["STUB", "TODO", "PLACEHOLDER", "# Replace", "would be generated"]
    is_stub = any(marker in generated_code.upper() for marker in stub_markers)
    is_empty = not generated_code or not generated_code.strip() or len(generated_code.strip()) < 50
    
    if is_stub or is_empty:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(
            f"Generated code is stub or empty (stub={is_stub}, empty={is_empty}). "
            "Code generation must produce valid simulation code."
        )
        # Increment revision count and provide feedback
        result = {
            "workflow_phase": "code_generation",
            "code": generated_code,
            "code_revision_count": state.get("code_revision_count", 0) + 1,
            "reviewer_feedback": (
                "ERROR: Generated code is empty or contains stub markers. "
                "Code generation must produce valid Meep simulation code. "
                "Please regenerate with proper implementation."
            ),
        }
        return result
    
    result = {
        "workflow_phase": "code_generation",
        "code": generated_code
        # agent_output fields would go here (simulation_code, etc.)
    }
    
    log_agent_call("CodeGeneratorAgent", "generate_code", start_time)(state, result)
    return result


def execution_validator_node(state: ReproState) -> dict:
    """
    ExecutionValidatorAgent: Validate simulation ran correctly.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: 
    - Sets `execution_verdict` state field from agent output's `verdict`.
    - Increments `execution_failure_count` when verdict is "fail".
    - Increments `total_execution_failures` for metrics tracking.
    The routing function `route_after_execution_check` reads these fields.
    """
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT CHECK: Execution analysis may include long logs
    # ═══════════════════════════════════════════════════════════════════════
    context_update = _check_context_or_escalate(state, "execution_check")
    if context_update:
        if context_update.get("awaiting_user_input"):
            return context_update
        state = {**state, **context_update}

    # Connect prompt adaptation
    system_prompt = build_agent_prompt("execution_validator", state)
    
    # Inject run_error into prompt if present
    run_error = state.get("run_error")
    if run_error:
        system_prompt += f"\n\nCONTEXT: The previous execution failed with error:\n{run_error}\nAnalyze this error and suggest fixes."
    
    # TODO: Implement execution validation logic
    # - Check completion status
    # - Verify output files exist
    # - Check for NaN/Inf in data
    # - Check for TIMEOUT_ERROR in run_error
    # - Call LLM with execution_validator_agent.md prompt
    # - Parse agent output per execution_validator_output_schema.json
    
    run_error = state.get("run_error")
    
    # Check fallback strategy if we are about to fail
    from schemas.state import get_stage_design_spec
    fallback = get_stage_design_spec(state, state.get("current_stage_id"), "fallback_strategy", "ask_user")
    
    if run_error and "TIMEOUT_ERROR" in run_error:
        if fallback == "skip_with_warning":
             agent_output = {
                "verdict": "pass", # Technically pass to move on, but status will be blocked/partial
                "stage_id": state.get("current_stage_id"),
                "summary": f"Execution timed out (skip_with_warning): {run_error}",
            }
        else:
            # Auto-fail on timeout if not handled by LLM
            agent_output = {
                "verdict": "fail",
                "stage_id": state.get("current_stage_id"),
                "summary": f"Execution timed out: {run_error}",
            }
    else:
        # STUB: Replace with actual LLM call
        agent_output = {
            "verdict": "pass",  # "pass" | "warning" | "fail"
            "stage_id": state.get("current_stage_id"),
            "summary": "Execution validation stub - implement with LLM call",
        }
    
    result = {
        "workflow_phase": "execution_validation",
        # Copy verdict to type-specific state field for routing
        "execution_verdict": agent_output["verdict"],
    }
    
    # Increment failure counters if verdict is "fail"
    # This happens BEFORE routing function reads the count
    if agent_output["verdict"] == "fail":
        result["execution_failure_count"] = state.get("execution_failure_count", 0) + 1
        result["total_execution_failures"] = state.get("total_execution_failures", 0) + 1
        
        runtime_config = state.get("runtime_config", {})
        max_failures = runtime_config.get("max_execution_failures", MAX_EXECUTION_FAILURES)
        current_failures = state.get("execution_failure_count", 0)
        
        if (current_failures + 1) >= max_failures:
            result["ask_user_trigger"] = "execution_failure_limit"
            result["pending_user_questions"] = [
                f"Execution failed {max_failures} times. Last error: {run_error or 'Unknown'}. "
                "Options: RETRY_WITH_GUIDANCE (provide hint), SKIP_STAGE, or STOP?"
            ]
    
    return result


def physics_sanity_node(state: ReproState) -> dict:
    """
    PhysicsSanityAgent: Validate physics of results.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: 
    - Sets `physics_verdict` state field from agent output's `verdict`.
    - Increments `physics_failure_count` when verdict is "fail".
    - Increments `design_revision_count` when verdict is "design_flaw".
    The routing function `route_after_physics_check` reads these fields.
    
    Verdict options:
    - "pass": Physics looks good, proceed to analysis
    - "warning": Minor concerns but proceed
    - "fail": Code/numerics issue, route to code generator
    - "design_flaw": Fundamental design problem, route to simulation designer
    """
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT CHECK: Physics sanity can include large data payloads
    # ═══════════════════════════════════════════════════════════════════════
    context_update = _check_context_or_escalate(state, "physics_check")
    if context_update:
        if context_update.get("awaiting_user_input"):
            return context_update
        state = {**state, **context_update}

    # Connect prompt adaptation
    system_prompt = build_agent_prompt("physics_sanity", state)
    
    # TODO: Implement physics validation logic
    # - Check conservation laws (T + R + A ≈ 1)
    # - Verify value ranges
    # - Check numerical quality
    # - Call LLM with physics_sanity_agent.md prompt
    # - Parse agent output per physics_sanity_output_schema.json
    
    # STUB: Replace with actual LLM call
    agent_output = {
        "verdict": "pass",  # "pass" | "warning" | "fail" | "design_flaw"
        "stage_id": state.get("current_stage_id"),
        "summary": "Physics validation stub - implement with LLM call",
        "backtrack_suggestion": {"suggest_backtrack": False},
    }
    
    result = {
        "workflow_phase": "physics_validation",
        # Copy verdict to type-specific state field for routing
        "physics_verdict": agent_output["verdict"],
    }
    
    # Increment failure counters based on verdict type
    # This happens BEFORE routing function reads the count
    if agent_output["verdict"] == "fail":
        result["physics_failure_count"] = state.get("physics_failure_count", 0) + 1
    elif agent_output["verdict"] == "design_flaw":
        # design_flaw routes to design node, use design_revision_count
        result["design_revision_count"] = state.get("design_revision_count", 0) + 1
    
    # If agent suggests backtrack, populate backtrack_suggestion for supervisor
    if agent_output.get("backtrack_suggestion", {}).get("suggest_backtrack"):
        result["backtrack_suggestion"] = agent_output["backtrack_suggestion"]
    
    return result


def results_analyzer_node(state: ReproState) -> dict:
    """ResultsAnalyzerAgent: Compare results to paper figures."""
    context_update = _check_context_or_escalate(state, "analyze")
    if context_update:
        if context_update.get("awaiting_user_input"):
            return context_update  # type: ignore[return-value]
        state = {**state, **context_update}
    
    state["workflow_phase"] = "analysis"
    current_stage_id = state.get("current_stage_id")
    stage_info = get_plan_stage(state, current_stage_id) if current_stage_id else None
    figures = _ensure_stub_figures(state)
    target_ids = []
    if stage_info and stage_info.get("targets"):
        target_ids = stage_info["targets"]
    elif stage_info and stage_info.get("target_details"):
        target_ids = [t.get("figure_id") for t in stage_info["target_details"] if t.get("figure_id")]
    else:
        target_ids = [fig.get("id", "FigStub") for fig in figures]
    
    feedback_targets = _extract_targets_from_feedback(state.get("analysis_feedback"), target_ids)
    ordered_targets = feedback_targets + [t for t in target_ids if t not in feedback_targets]
    ordered_targets = ordered_targets or target_ids
    
    # ═══════════════════════════════════════════════════════════════════════
    # VALIDATE STAGE OUTPUTS: Ensure outputs exist before analysis
    # ═══════════════════════════════════════════════════════════════════════
    stage_outputs = state.get("stage_outputs", {})
    output_files = stage_outputs.get("files", [])
    
    if not stage_outputs or not output_files:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(
            f"Stage outputs are empty or missing for stage {current_stage_id}. "
            "Cannot proceed with analysis without simulation outputs."
        )
        # Route back to execution validation with error
        return {
            "workflow_phase": "analysis",
            "execution_verdict": "fail",
            "run_error": (
                f"Stage outputs are missing for {current_stage_id}. "
                "Simulation may not have completed successfully. "
                "Please check execution logs and rerun simulation."
            ),
            "analysis_summary": "Analysis skipped: No outputs available",
        }
    
    # Validate that output files actually exist on disk
    from pathlib import Path
    existing_files = []
    missing_files = []
    for file_path in output_files:
        file_obj = Path(file_path) if isinstance(file_path, str) else Path(str(file_path))
        if file_obj.exists() and file_obj.is_file():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    if not existing_files and output_files:
        # All files are missing - this is an error
        import logging
        logger = logging.getLogger(__name__)
        logger.error(
            f"All output files are missing from disk: {output_files}. "
            "Files may have been deleted or simulation failed to write outputs."
        )
        return {
            "workflow_phase": "analysis",
            "execution_verdict": "fail",
            "run_error": (
                f"Output files listed in stage_outputs do not exist on disk: {missing_files}. "
                "Simulation may have failed to write outputs or files were deleted."
            ),
            "analysis_summary": "Analysis skipped: Output files missing",
        }
    
    # Use only existing files for analysis
    if missing_files:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Some output files are missing: {missing_files}. "
            f"Proceeding with {len(existing_files)} available files."
        )
        output_files = existing_files
    paper_id = state.get("paper_id", "paper")
    expected_outputs_map = _collect_expected_outputs(stage_info, paper_id, current_stage_id or "")
    plan_stage_columns = _collect_expected_columns(stage_info)
    figure_lookup = {fig.get("id") or fig.get("figure_id"): fig for fig in figures}
    if stage_info and stage_info.get("target_details"):
        target_meta_list = stage_info["target_details"]
    else:
        target_meta_list = state.get("plan", {}).get("targets", [])
    plan_targets_map = {target.get("figure_id"): target for target in target_meta_list}
    matched_targets: List[str] = []
    pending_targets: List[str] = []
    missing_targets: List[str] = []
    mismatch_targets: List[str] = []
    stage_discrepancies: List[Dict[str, Any]] = []
    
    figure_comparisons: List[Dict[str, Any]] = []
    per_result_reports: List[Dict[str, Any]] = []
    
    for target_id in ordered_targets:
        expected_names = expected_outputs_map.get(target_id, [])
        matched_file = _match_expected_files(expected_names, output_files)
        if not matched_file:
            matched_file = _match_output_file(output_files, target_id)
        has_output = matched_file is not None
        figure_meta = figure_lookup.get(target_id) or {}
        target_cfg = plan_targets_map.get(target_id, {})
        precision_requirement = target_cfg.get("precision_requirement", "acceptable")
        reference_path = target_cfg.get("reference_data_path") or (stage_info.get("reference_data_path") if stage_info else None)
        digitized_path = (
            figure_meta.get("digitized_data_path")
            or target_cfg.get("digitized_data_path")
            or target_cfg.get("digitized_reference")
            or reference_path
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # VALIDATE FIGURE IMAGE PATH: Check file exists before use
        # ═══════════════════════════════════════════════════════════════════════
        paper_image_path = figure_meta.get("image_path")
        if paper_image_path:
            from pathlib import Path
            image_file = Path(paper_image_path)
            if not image_file.exists() or not image_file.is_file():
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Figure image path does not exist or is not a file: {paper_image_path}. "
                    f"Comparison for {target_id} will proceed without reference image."
                )
                # Mark as missing reference but continue processing
                paper_image_path = None
        requires_digitized = precision_requirement == "excellent"
        quantitative_metrics: Dict[str, Any] = {}
        classification_label = "missing_output"
        
        if not has_output:
            missing_targets.append(target_id)
            stage_discrepancies.append(
                _record_discrepancy(
                    state,
                    current_stage_id,
                    target_id,
                    "output_artifact",
                    target_cfg.get("description", "Expected artifact generated per plan"),
                    "Not generated",
                    classification="investigate",
                    likely_cause="Simulation run did not create expected output file.",
                    action_taken="Flagged for analyzer follow-up",
                    blocking=True,
                )
            )
        else:
            expected_columns = plan_stage_columns.get(target_id)
            sim_series = _load_numeric_series(matched_file, expected_columns)
            ref_series = _load_numeric_series(digitized_path, expected_columns)
            quantitative_metrics = _quantitative_curve_metrics(sim_series, ref_series)
            has_reference = ref_series is not None
            
            if requires_digitized and not digitized_path:
                classification_label = "pending_validation"
                pending_targets.append(target_id)
                stage_discrepancies.append(
                    _record_discrepancy(
                        state,
                        current_stage_id,
                        target_id,
                        "digitized_reference",
                        "Digitized reference required",
                        "Not provided",
                        classification="investigate",
                        likely_cause="precision_requirement='excellent' but digitized data missing",
                        action_taken="Awaiting digitized data or user guidance",
                        blocking=False,
                    )
                )
            else:
                classification_label = _classification_from_metrics(
                    quantitative_metrics,
                    precision_requirement,
                    has_reference,
                )
                if classification_label == "match":
                    matched_targets.append(target_id)
                elif classification_label in {"pending_validation", "partial_match"}:
                    pending_targets.append(target_id)
                else:
                    mismatch_targets.append(target_id)
                
                error_percent = quantitative_metrics.get("peak_position_error_percent")
                if error_percent is not None and classification_label in {"partial_match", "mismatch"}:
                    discrepancy_class = "acceptable" if classification_label == "partial_match" else "investigate"
                    blocking = classification_label == "mismatch"
                    paper_peak = quantitative_metrics.get("peak_position_paper")
                    sim_peak = quantitative_metrics.get("peak_position_sim")
                    stage_discrepancies.append(
                        _record_discrepancy(
                            state,
                            current_stage_id,
                            target_id,
                            "resonance_wavelength",
                            f"{paper_peak:.2f} nm" if paper_peak else "Paper peak unavailable",
                            f"{sim_peak:.2f} nm" if sim_peak else "Simulation peak unavailable",
                            classification=discrepancy_class,
                            difference_percent=error_percent,
                            likely_cause="See analyzer notes",
                            action_taken="Documented for supervisor review",
                            blocking=blocking,
                        )
                    )
        
        comparison_table = [{
            "feature": "Simulation Output",
            "paper": figure_meta.get("description", target_id),
            "reproduction": matched_file or "Not generated",
            "status": (
                "✅ Match" if has_output and classification_label == "match"
                else "⚠️ Pending" if classification_label in {"pending_validation", "partial_match"}
                else "❌ Missing"
            ),
        }]
        if digitized_path:
            comparison_table.append({
                "feature": "Digitized reference",
                "paper": digitized_path,
                "reproduction": matched_file or "Not generated",
                "status": (
                    "Pending review" if classification_label not in {"missing_output", "mismatch"}
                    else "❌ Missing"
                ),
            })
        
        comparison_entry = {
            "figure_id": target_id,
            "stage_id": current_stage_id,
            "title": figure_meta.get("description", target_id),
            "paper_image_path": figure_meta.get("image_path"),
            "reproduction_image_path": matched_file,
            "comparison_table": comparison_table,
            "shape_comparison": [],
            "reason_for_difference": "" if has_output else "No simulation output matched this figure.",
            "classification": classification_label,
        }
        figure_comparisons.append(comparison_entry)
        
        target_criteria = []
        if stage_info:
            for criterion in stage_info.get("validation_criteria", []):
                if target_id and target_id.lower() in criterion.lower():
                    target_criteria.append(criterion)
        criteria_passed, criteria_failures = _evaluate_validation_criteria(quantitative_metrics, target_criteria)
        if target_criteria and not criteria_failures and not quantitative_metrics:
            criteria_failures.append("Validation criteria defined but quantitative metrics missing.")
        if criteria_failures:
            if target_id not in mismatch_targets:
                mismatch_targets.append(target_id)
            classification_label = "mismatch"
            for failure in criteria_failures:
                sim_value_display = json.dumps(quantitative_metrics) if quantitative_metrics else "N/A"
                stage_discrepancies.append(
                    _record_discrepancy(
                        state,
                        current_stage_id,
                        target_id,
                        "validation_criteria",
                        failure,
                        sim_value_display,
                        classification="investigate",
                        likely_cause="Validation criterion not satisfied",
                        action_taken="Flagged for revision",
                        blocking=True,
                    )
                )
        
        per_result_reports.append({
            "result_id": f"{current_stage_id or 'stage'}_{target_id}",
            "target_figure": target_id,
            "status": classification_label,
            "expected_outputs": expected_names,
            "matched_output": matched_file,
            "precision_requirement": precision_requirement,
            "digitized_data_path": digitized_path,
            "validation_criteria": target_criteria,
            "quantitative_metrics": quantitative_metrics,
             "criteria_failures": criteria_failures,
            "notes": "Output identified." if has_output else "Output missing; requires rerun.",
        })
    
    total_targets = len(target_ids)
    missing_count = len(missing_targets)
    pending_count = len(pending_targets)
    mismatch_count = len(mismatch_targets)
    
    if total_targets == 0:
        overall_classification = "NO_TARGETS"
    elif missing_count > 0 or mismatch_count > 0:
        overall_classification = "POOR_MATCH"
    elif pending_count > 0:
        overall_classification = "PARTIAL_MATCH"
    elif len(matched_targets) == total_targets:
        overall_classification = "EXCELLENT_MATCH"
    else:
        overall_classification = "ACCEPTABLE_MATCH"
    
    summary_notes_parts = []
    if state.get("analysis_feedback"):
        summary_notes_parts.append(f"Validator feedback: {state['analysis_feedback']}")
    if total_targets:
        summary_notes_parts.append(f"{len(matched_targets)}/{total_targets} targets currently classified as matches.")
    else:
        summary_notes_parts.append("No explicit targets defined for this stage.")
    summary_notes = " ".join(summary_notes_parts)
    
    summary = {
        "stage_id": current_stage_id,
        "totals": {
            "targets": total_targets,
            "matches": len(matched_targets),
            "pending": pending_count,
            "missing": missing_count,
            "mismatch": mismatch_count,
        },
        "matched_targets": matched_targets,
        "pending_targets": pending_targets,
        "missing_targets": missing_targets,
        "mismatch_targets": mismatch_targets,
        "discrepancies_logged": len(stage_discrepancies),
        "validation_criteria": stage_info.get("validation_criteria", []) if stage_info else [],
        "feedback_applied": feedback_targets,
        "unresolved_targets": missing_targets + pending_targets + mismatch_targets,
        "notes": summary_notes,
    }
    
    existing_comparisons = state.get("figure_comparisons", [])
    filtered_existing = [
        comp for comp in existing_comparisons
        if comp.get("stage_id") != current_stage_id
    ]
    
    existing_reports = state.get("analysis_result_reports", [])
    reports_with_stage = [
        {**report, "stage_id": current_stage_id}
        for report in per_result_reports
    ]
    filtered_reports = [
        report for report in existing_reports
        if report.get("stage_id") != current_stage_id
    ]
    merged_reports = filtered_reports + reports_with_stage
    
    unresolved = summary["unresolved_targets"]
    analysis_feedback_next = None if not unresolved else state.get("analysis_feedback")
    
    return {
        "workflow_phase": "analysis",
        "analysis_summary": summary,
        "analysis_overall_classification": overall_classification,
        "analysis_result_reports": merged_reports,
        "figure_comparisons": filtered_existing + figure_comparisons,
        "analysis_feedback": analysis_feedback_next,
    }


def comparison_validator_node(state: ReproState) -> dict:
    """
    ComparisonValidatorAgent: Validate comparison accuracy.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: 
    - Sets `comparison_verdict` state field from agent output's `verdict`.
    - Increments `analysis_revision_count` when verdict is "needs_revision".
    The routing function `route_after_comparison_check` reads these fields.
    """
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT CHECK: Comparison validator reads analyzer outputs
    # ═══════════════════════════════════════════════════════════════════════
    context_update = _check_context_or_escalate(state, "comparison_check")
    if context_update:
        if context_update.get("awaiting_user_input"):
            return context_update
        state = {**state, **context_update}

    stage_id = state.get("current_stage_id")
    comparisons = _stage_comparisons_for_stage(state, stage_id)
    breakdown = _breakdown_comparison_classifications(comparisons)
    stage_info = get_plan_stage(state, stage_id) if stage_id else None
    if stage_info:
        expected_targets = stage_info.get("targets") or [
            t.get("figure_id") for t in stage_info.get("target_details", []) if t.get("figure_id")
        ]
    else:
        expected_targets = []
    analysis_reports = _analysis_reports_for_stage(state, stage_id)
    report_issues = _validate_analysis_reports(analysis_reports)
    missing_report_targets = [
        target for target in expected_targets
        if target not in {report.get("target_figure") for report in analysis_reports}
    ]
    
    if not comparisons:
        if not expected_targets:
            verdict = "approve"
            feedback = "Stage has no reproducible targets; nothing to compare."
        else:
            verdict = "needs_revision"
            feedback = "Results analyzer did not produce figure comparisons for this stage."
    elif breakdown["missing"]:
        verdict = "needs_revision"
        feedback = f"Simulation outputs missing for: {', '.join(breakdown['missing'])}"
    elif breakdown["pending"]:
        verdict = "needs_revision"
        feedback = f"Comparisons pending quantitative checks for: {', '.join(breakdown['pending'])}"
    else:
        verdict = "approve"
        feedback = "All required comparisons present."
    
    missing_comparisons = [
        target for target in expected_targets
        if target not in {comp.get("figure_id") for comp in comparisons}
    ]
    if missing_comparisons:
        verdict = "needs_revision"
        feedback = f"Results analyzer did not produce comparisons for: {', '.join(missing_comparisons)}"
    
    if report_issues or missing_report_targets:
        verdict = "needs_revision"
        combined = report_issues + (
            [f"Missing quantitative reports for: {', '.join(missing_report_targets)}"]
            if missing_report_targets else []
        )
        feedback = "; ".join(combined[:3])
        if len(combined) > 3:
            feedback += f" (+{len(combined)-3} more)"
    result = {
        "workflow_phase": "comparison_validation",
        "comparison_verdict": verdict,
        "comparison_feedback": feedback,
    }
    
    if verdict == "needs_revision":
        result["analysis_revision_count"] = state.get("analysis_revision_count", 0) + 1
        result["analysis_feedback"] = feedback
    else:
        # Clear feedback on success
        result["analysis_feedback"] = None
    
    return result


def supervisor_node(state: ReproState) -> dict:
    """
    SupervisorAgent: Big-picture assessment and decisions.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT implementation notes:
    
    1. CHECK ask_user_trigger:
       When this node is called after ask_user, check state["ask_user_trigger"]
       to understand what the user was responding to. See src/prompts.py
       ASK_USER_TRIGGERS for full documentation of all trigger types.
       
    2. HANDLE EACH TRIGGER TYPE:
       Each trigger has specific handling requirements documented in
       src/prompts.py:ASK_USER_TRIGGERS. Key triggers:
       - "material_checkpoint": Mandatory Stage 0 validation
       - "code_review_limit": User guidance on stuck code generation
       - "design_review_limit": User guidance on stuck design
       - "execution_failure_limit": Simulation runtime failures
       - "physics_failure_limit": Physics sanity check failures
       - "context_overflow": LLM context management
       - "replan_limit": Planning iteration limit
       - "backtrack_approval": Cross-stage backtrack confirmation
       
    3. RESET COUNTERS on user intervention:
       When user provides guidance that resolves an issue, reset relevant
       counters to prevent limit exhaustion:
       - code_revision_count = 0 if routing back to generate_code
       - design_revision_count = 0 if routing back to design
       - execution_failure_count = 0 if retrying execution
       - physics_failure_count = 0 if retrying physics check
       
    4. USE get_validation_hierarchy():
       Always use get_validation_hierarchy(state) to check hierarchy status.
       Never store validation_hierarchy in state directly - it's computed.
    """
    from schemas.state import (
        get_validation_hierarchy, 
        update_progress_stage_status,
        archive_stage_outputs_to_progress
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT CHECK: Supervisor aggregates most of the state
    # ═══════════════════════════════════════════════════════════════════════
    context_update = _check_context_or_escalate(state, "supervisor")
    if context_update:
        if context_update.get("awaiting_user_input"):
            return context_update
        state = {**state, **context_update}

    def _derive_stage_completion_outcome(current_state: ReproState, stage_id: Optional[str]) -> Tuple[str, str]:
        classification = (current_state.get("analysis_overall_classification") or "").upper()
        comparison_verdict = current_state.get("comparison_verdict")
        physics_verdict = current_state.get("physics_verdict")
        comparisons = _stage_comparisons_for_stage(current_state, stage_id)
        comparison_breakdown = _breakdown_comparison_classifications(comparisons)
        
        classification_map = {
            "FAILED": "completed_failed",
            "POOR_MATCH": "completed_failed",
            "PARTIAL_MATCH": "completed_partial",
            "ACCEPTABLE_MATCH": "completed_success",
            "EXCELLENT_MATCH": "completed_success",
            "NO_TARGETS": "completed_success",
        }
        status = classification_map.get(classification, "completed_success")
        
        if comparison_breakdown["missing"]:
            status = "completed_failed"
        elif comparison_breakdown["pending"] and status == "completed_success":
            status = "completed_partial"
        
        if comparison_verdict == "needs_revision":
            status = "completed_partial"
        if physics_verdict == "warning" and status == "completed_success":
            status = "completed_partial"
        if physics_verdict == "fail" or classification == "":
            if physics_verdict == "fail":
                status = "completed_failed"
        
        summary_data = current_state.get("analysis_summary")
        if comparison_breakdown["missing"]:
            summary_text = f"Missing outputs for: {', '.join(comparison_breakdown['missing'])}"
        elif comparison_breakdown["pending"]:
            summary_text = f"Comparisons pending for: {', '.join(comparison_breakdown['pending'])}"
        elif isinstance(summary_data, dict):
            totals = summary_data.get("totals", {})
            summary_text = summary_data.get("notes") or \
                f"{totals.get('matches', 0)}/{totals.get('targets', 0)} targets matched"
        else:
            summary_text = summary_data or \
                f"Stage classified as {classification or comparison_verdict or 'OK_CONTINUE'}"
        
        return status, summary_text

    # Connect prompt adaptation
    system_prompt = build_agent_prompt("supervisor", state)
    
    # TODO: Implement full supervision logic using prompts/supervisor_agent.md
    # - Call LLM with supervisor_agent.md prompt
    # - Parse agent output per supervisor_output_schema.json
    
    # Get trigger info
    ask_user_trigger = state.get("ask_user_trigger")
    user_responses = state.get("user_responses", {})
    last_node = state.get("last_node_before_ask_user")
    current_stage_id = state.get("current_stage_id")
    
    result: Dict[str, Any] = {
        "workflow_phase": "supervision",
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # POST-ASK_USER HANDLING: Route based on trigger type
    # ═══════════════════════════════════════════════════════════════════════
    
    if ask_user_trigger:
        result["ask_user_trigger"] = None  # Clear trigger after handling
        
        # ─── MATERIAL CHECKPOINT ──────────────────────────────────────────────
        if ask_user_trigger == "material_checkpoint":
            # Parse user response from any question (get last response)
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            # Explicit approval keywords
            approval_keywords = ["APPROVE", "YES", "CORRECT", "OK", "ACCEPT", "VALID", "PROCEED"]
            rejection_keywords = ["REJECT", "NO", "WRONG", "INCORRECT", "CHANGE", "FIX"]
            
            is_approval = any(kw in response_text for kw in approval_keywords)
            is_rejection = any(kw in response_text for kw in rejection_keywords)
            
            if is_approval and not is_rejection:
                # Explicit approval - proceed with materials
                result["supervisor_verdict"] = "ok_continue"
                result["supervisor_feedback"] = "Material validation approved by user."
                # CRITICAL: Move pending materials to validated_materials on approval
                pending_materials = state.get("pending_validated_materials", [])
                if pending_materials:
                    result["validated_materials"] = pending_materials
                    result["pending_validated_materials"] = []  # Clear pending
                else:
                    # No materials to validate - this is an error state
                    result["supervisor_verdict"] = "ask_user"
                    result["pending_user_questions"] = [
                        "ERROR: No materials were extracted for validation. "
                        "Please specify materials manually using CHANGE_MATERIAL or CHANGE_DATABASE."
                    ]
                    return result
                
                # Archive outputs before moving on
                if current_stage_id:
                    try:
                        archive_stage_outputs_to_progress(state, current_stage_id)
                    except Exception as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(
                            f"Failed to archive outputs for stage {current_stage_id}: {e}. "
                            "Continuing but outputs may not be persisted."
                        )
                        # Set flag for potential retry later
                        if "archive_errors" not in state:
                            state["archive_errors"] = []
                        state["archive_errors"].append({
                            "stage_id": current_stage_id,
                            "error": str(e),
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                    update_progress_stage_status(state, current_stage_id, "completed_success")
                    
            elif "CHANGE_DATABASE" in response_text or (is_rejection and "DATABASE" in response_text):
                # User requested database change - requires REPLAN to update assumptions
                result["supervisor_verdict"] = "replan_needed"
                result["planner_feedback"] = (
                    f"User rejected material validation and requested database change: {response_text}. "
                    "Please update the plan and assumptions to use the specified database/material, "
                    "then re-run Stage 0."
                )
                # Mark stage 0 as invalid
                if current_stage_id:
                    update_progress_stage_status(
                        state, 
                        current_stage_id, 
                        "invalidated", 
                        invalidation_reason="User requested material change"
                    )
                # Clear pending materials since they're rejected
                result["pending_validated_materials"] = []
                # ═══════════════════════════════════════════════════════════════════════
                # CLEAR VALIDATED_MATERIALS: Remove any previously validated materials
                # ═══════════════════════════════════════════════════════════════════════
                result["validated_materials"] = []  # Clear validated materials on rejection
            elif "CHANGE_MATERIAL" in response_text or (is_rejection and "MATERIAL" in response_text):
                # Need to replan with different material
                result["supervisor_verdict"] = "replan_needed"
                result["planner_feedback"] = f"User indicated wrong material: {response_text}. Please update plan."
                # Mark stage 0 as invalid
                if current_stage_id:
                    update_progress_stage_status(
                        state,
                        current_stage_id,
                        "invalidated",
                        invalidation_reason="User rejected material"
                    )
                # Clear pending materials since they're rejected
                result["pending_validated_materials"] = []
                # ═══════════════════════════════════════════════════════════════════════
                # CLEAR VALIDATED_MATERIALS: Remove any previously validated materials
                # ═══════════════════════════════════════════════════════════════════════
                result["validated_materials"] = []  # Clear validated materials on rejection
            elif "NEED_HELP" in response_text or "HELP" in response_text:
                result["supervisor_verdict"] = "ask_user"
                # OVERWRITE pending questions to avoid loop
                result["pending_user_questions"] = [
                    "Please provide more details about the material issue. "
                    "What specific aspect of the optical constants looks incorrect? "
                    "(e.g., 'Imaginary part too high', 'Wrong units', 'Resonance missing'). "
                    "Or respond with APPROVE, CHANGE_MATERIAL, or CHANGE_DATABASE."
                ]
            elif is_rejection:
                # Generic rejection - ask for clarification
                result["supervisor_verdict"] = "ask_user"
                result["pending_user_questions"] = [
                    "You indicated rejection but didn't specify what to change. "
                    "Please respond with: APPROVE (to accept materials), "
                    "CHANGE_MATERIAL (to specify different material), or "
                    "CHANGE_DATABASE (to use different optical constants database)."
                ]
            else:
                # Ambiguous response - ask for clarification instead of assuming approval
                result["supervisor_verdict"] = "ask_user"
                result["pending_user_questions"] = [
                    f"Your response '{response_text[:100]}' is unclear. "
                    "Please respond with: APPROVE (to accept materials), "
                    "CHANGE_MATERIAL (to specify different material), or "
                    "CHANGE_DATABASE (to use different optical constants database)."
                ]
        
        # ─── CODE REVIEW LIMIT ────────────────────────────────────────────────
        elif ask_user_trigger == "code_review_limit":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "PROVIDE_HINT" in response_text or "HINT" in response_text:
                # Reset counter and route back to code generation with hint
                result["code_revision_count"] = 0
                result["reviewer_feedback"] = f"User hint: {user_responses.get(list(user_responses.keys())[-1] if user_responses else '', '')}"
                result["supervisor_verdict"] = "ok_continue"  # Will route to select_stage, then design->code
                result["supervisor_feedback"] = "Retrying code generation with user hint."
            elif "SKIP" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                if current_stage_id:
                    update_progress_stage_status(state, current_stage_id, "blocked", 
                                                summary="Skipped by user due to code review issues")
            elif "STOP" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
            else:
                result["supervisor_verdict"] = "ask_user"
                result["pending_user_questions"] = [
                    "Please clarify: PROVIDE_HINT (with hint text), SKIP_STAGE, or STOP?"
                ]
        
        # ─── DESIGN REVIEW LIMIT ──────────────────────────────────────────────
        elif ask_user_trigger == "design_review_limit":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "PROVIDE_HINT" in response_text or "HINT" in response_text:
                result["design_revision_count"] = 0
                result["reviewer_feedback"] = f"User hint: {user_responses.get(list(user_responses.keys())[-1] if user_responses else '', '')}"
                result["supervisor_verdict"] = "ok_continue"
            elif "SKIP" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                if current_stage_id:
                    update_progress_stage_status(state, current_stage_id, "blocked",
                                                summary="Skipped by user due to design review issues")
            elif "STOP" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
            else:
                result["supervisor_verdict"] = "ask_user"
                result["pending_user_questions"] = [
                    "Please clarify: PROVIDE_HINT (with hint text), SKIP_STAGE, or STOP?"
                ]
        
        # ─── EXECUTION FAILURE LIMIT ──────────────────────────────────────────
        elif ask_user_trigger == "execution_failure_limit":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "RETRY" in response_text or "GUIDANCE" in response_text:
                result["execution_failure_count"] = 0
                result["supervisor_feedback"] = f"User guidance: {user_responses.get(list(user_responses.keys())[-1] if user_responses else '', '')}"
                result["supervisor_verdict"] = "ok_continue"
            elif "SKIP" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                if current_stage_id:
                    update_progress_stage_status(state, current_stage_id, "blocked",
                                                summary="Skipped by user due to execution failures")
            elif "STOP" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
            else:
                result["supervisor_verdict"] = "ask_user"
                result["pending_user_questions"] = [
                    "Please clarify: RETRY_WITH_GUIDANCE (with guidance), SKIP_STAGE, or STOP?"
                ]
        
        # ─── PHYSICS FAILURE LIMIT ────────────────────────────────────────────
        elif ask_user_trigger == "physics_failure_limit":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "RETRY" in response_text:
                result["physics_failure_count"] = 0
                result["supervisor_feedback"] = f"User guidance: {user_responses.get(list(user_responses.keys())[-1] if user_responses else '', '')}"
                result["supervisor_verdict"] = "ok_continue"
            elif "ACCEPT" in response_text or "PARTIAL" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                if current_stage_id:
                    update_progress_stage_status(state, current_stage_id, "completed_partial",
                                                summary="Accepted as partial by user despite physics issues")
            elif "SKIP" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                if current_stage_id:
                    update_progress_stage_status(state, current_stage_id, "blocked",
                                                summary="Skipped by user due to physics check failures")
            elif "STOP" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
            else:
                result["supervisor_verdict"] = "ask_user"
                result["pending_user_questions"] = [
                    "Please clarify: RETRY_WITH_GUIDANCE, ACCEPT_PARTIAL, SKIP_STAGE, or STOP?"
                ]
        
        # ─── CONTEXT OVERFLOW ─────────────────────────────────────────────────
        elif ask_user_trigger == "context_overflow":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "SUMMARIZE" in response_text:
                # Apply feedback summarization (handled elsewhere)
                result["supervisor_verdict"] = "ok_continue"
                result["supervisor_feedback"] = "Applying feedback summarization for context management."
            elif "TRUNCATE" in response_text:
                # Actually truncate the paper text to resolve the loop
                current_text = state.get("paper_text", "")
                # Keep first 15k chars and last 5k as a simple heuristic
                if len(current_text) > 20000:
                    truncated_text = current_text[:15000] + "\n\n... [TRUNCATED BY USER REQUEST] ...\n\n" + current_text[-5000:]
                    result["paper_text"] = truncated_text
                    result["supervisor_feedback"] = "Truncating paper to first 15k and last 5k chars."
                else:
                    result["supervisor_feedback"] = "Paper already short enough, proceeding."
                
                result["supervisor_verdict"] = "ok_continue"
            elif "SKIP" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                if current_stage_id:
                    update_progress_stage_status(state, current_stage_id, "blocked",
                                                summary="Skipped due to context overflow")
            elif "STOP" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
            else:
                result["supervisor_verdict"] = "ok_continue"
        
        # ─── REPLAN LIMIT ─────────────────────────────────────────────────────
        elif ask_user_trigger == "replan_limit":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "FORCE" in response_text or "ACCEPT" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                result["supervisor_feedback"] = "Plan force-accepted by user."
            elif "GUIDANCE" in response_text:
                result["replan_count"] = 0
                result["planner_feedback"] = f"User guidance: {user_responses.get(list(user_responses.keys())[-1] if user_responses else '', '')}"
                result["supervisor_verdict"] = "replan_needed"
            elif "STOP" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
            else:
                result["supervisor_verdict"] = "ok_continue"
        
        # ─── BACKTRACK APPROVAL ───────────────────────────────────────────────
        elif ask_user_trigger == "backtrack_approval":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "APPROVE" in response_text:
                result["supervisor_verdict"] = "backtrack_to_stage"
                
                # Ensure backtrack_decision has stages_to_invalidate populated
                decision = state.get("backtrack_decision", {})
                if decision:
                    target = decision.get("target_stage_id")
                    if target:
                        dependent = _get_dependent_stages(state.get("plan", {}), target)
                        decision["stages_to_invalidate"] = dependent
                        result["backtrack_decision"] = decision
                        
            elif "REJECT" in response_text:
                result["backtrack_suggestion"] = None
                result["supervisor_verdict"] = "ok_continue"
            else:
                result["supervisor_verdict"] = "ok_continue"
        
        # ─── DEADLOCK DETECTED ──────────────────────────────────────────────────
        elif ask_user_trigger == "deadlock_detected":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
            
            if "GENERATE_REPORT" in response_text or "REPORT" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
                result["supervisor_feedback"] = "Generating report with current results despite deadlock."
            elif "REPLAN" in response_text:
                result["supervisor_verdict"] = "replan_needed"
                result["planner_feedback"] = (
                    f"User requested replan due to deadlock: {response_text}. "
                    "Please review blocked stages and fix dependencies or validation issues."
                )
            elif "STOP" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
            else:
                result["supervisor_verdict"] = "ask_user"
                result["pending_user_questions"] = [
                    "Please clarify: GENERATE_REPORT (with current results), REPLAN (to fix blocked stages), or STOP?"
                ]
        
        # ─── BACKTRACK LIMIT ──────────────────────────────────────────────────
        elif ask_user_trigger == "backtrack_limit":
            response_text = ""
            for q, r in user_responses.items():
                response_text = r.upper() if isinstance(r, str) else str(r)
                
            if "STOP" in response_text:
                result["supervisor_verdict"] = "all_complete"
                result["should_stop"] = True
            elif "SKIP" in response_text:
                result["supervisor_verdict"] = "ok_continue"
                if current_stage_id:
                    update_progress_stage_status(state, current_stage_id, "blocked", 
                                                summary="Skipped due to backtrack limit")
            else:
                # Default or FORCE_CONTINUE
                # Reset backtrack count to give more tries? Or just proceed?
                # If we proceed without resetting, we'll hit the limit again next time.
                # So we must reset or bump the limit.
                state["backtrack_count"] = 0 # Reset for this stage/context
                result["supervisor_verdict"] = "backtrack_to_stage" # Try again?
                # Actually, if we are here, we just finished handle_backtrack.
                # If we want to continue backtracking, we need to allow it.
                # But handle_backtrack already did the work.
                # We just need to route to select_stage.
                result["supervisor_verdict"] = "select_stage"
                
        # ─── UNKNOWN/DEFAULT ──────────────────────────────────────────────────
        else:
            # Unknown trigger - default to continue
            result["supervisor_verdict"] = "ok_continue"
            result["supervisor_feedback"] = f"Handled unknown trigger: {ask_user_trigger}"
    
    # ═══════════════════════════════════════════════════════════════════════
    # NORMAL SUPERVISION (not post-ask_user)
    # ═══════════════════════════════════════════════════════════════════════
    else:
        # TODO: Implement actual LLM-based supervision logic
        # For now, default to continue
        result["supervisor_verdict"] = "ok_continue"
        
        # If completing a stage, archive outputs
        if current_stage_id:
            try:
                archive_stage_outputs_to_progress(state, current_stage_id)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(
                    f"Failed to archive outputs for stage {current_stage_id}: {e}. "
                    "Continuing but outputs may not be persisted."
                )
                # Set flag for potential retry later
                if "archive_errors" not in state:
                    state["archive_errors"] = []
                state["archive_errors"].append({
                    "stage_id": current_stage_id,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            status, summary_text = _derive_stage_completion_outcome(state, current_stage_id)
            update_progress_stage_status(state, current_stage_id, status, summary=summary_text)
    
    # Log user interaction if one just happened
    if ask_user_trigger and user_responses:
        # Record structured interaction log
        interaction_entry = {
            "id": f"U{len(state.get('progress', {}).get('user_interactions', [])) + 1}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "interaction_type": ask_user_trigger,
            "context": {
                "stage_id": current_stage_id,
                "agent": "SupervisorAgent",
                "reason": ask_user_trigger
            },
            "question": str(state.get("pending_user_questions", [""])[0]),
            "user_response": str(list(user_responses.values())[-1]),
            "impact": result.get("supervisor_feedback", "User decision processed"),
            "alternatives_considered": [] # Would be populated by LLM
        }
        
        # Ensure progress structure exists
        if "progress" not in state:
            state["progress"] = {}
        if "user_interactions" not in state["progress"]:
            state["progress"]["user_interactions"] = []
            
        state["progress"]["user_interactions"].append(interaction_entry)
    
    return result


# Helper function for dependency resolution
def _get_dependent_stages(plan: dict, target_stage_id: str) -> list:
    """
    Identify all stages that depend on the target stage (transitively).
    
    Args:
        plan: The plan dictionary containing stages and dependencies
        target_stage_id: The ID of the stage being backtracked to
        
    Returns:
        List of stage_ids that depend on target_stage_id
    """
    stages = plan.get("stages", [])
    # Build dependency graph: stage_id -> list of dependents
    dependents_map = {s["stage_id"]: [] for s in stages}
    
    for stage in stages:
        for dep in stage.get("dependencies", []):
            if dep in dependents_map:
                dependents_map[dep].append(stage["stage_id"])
                
    # BFS/DFS to find all transitive dependents
    invalidated = set()
    queue = [target_stage_id]
    
    while queue:
        current = queue.pop(0)
        if current in dependents_map:
            for dep in dependents_map[current]:
                if dep not in invalidated:
                    invalidated.add(dep)
                    queue.append(dep)
                    
    return list(invalidated)


def ask_user_node(state: ReproState) -> Dict[str, Any]:
    """
    CLI-based user interaction node.
    
    Prompts user in terminal for input. If user doesn't respond within timeout
    (Ctrl+C or timeout), saves checkpoint and exits gracefully.
    
    Environment variables:
        REPROLAB_USER_TIMEOUT_SECONDS: Override default timeout (default: 86400 = 24h)
        REPROLAB_NON_INTERACTIVE: If "1", immediately save checkpoint and exit
        
    Returns:
        Dict with state updates (user_responses, cleared pending questions)
    """
    timeout_seconds = int(os.environ.get("REPROLAB_USER_TIMEOUT_SECONDS", "86400"))
    non_interactive = os.environ.get("REPROLAB_NON_INTERACTIVE", "0") == "1"
    
    questions = state.get("pending_user_questions", [])
    trigger = state.get("ask_user_trigger", "unknown")
    paper_id = state.get("paper_id", "unknown")
    
    if not questions:
        return {
            "awaiting_user_input": False,
            "workflow_phase": "awaiting_user",
        }
    
    # Non-interactive mode: save and exit immediately
    if non_interactive:
        print("\n" + "=" * 60)
        print("USER INPUT REQUIRED (non-interactive mode)")
        print("=" * 60)
        print(f"\nPaper: {paper_id}")
        print(f"Trigger: {trigger}")
        for i, q in enumerate(questions, 1):
            print(f"\nQuestion {i}:\n{q}")
        
        checkpoint_path = save_checkpoint(state, f"awaiting_user_{trigger}")
        print(f"\n✓ Checkpoint saved: {checkpoint_path}")
        print("\nResume with:")
        print(f"  python -m src.graph --resume {checkpoint_path}")
        raise SystemExit(0)
    
    # Interactive mode: prompt user
    print("\n" + "=" * 60)
    print("USER INPUT REQUIRED")
    print("=" * 60)
    print(f"Paper: {paper_id}")
    print(f"Trigger: {trigger}")
    print("(Press Ctrl+C to save checkpoint and exit)")
    print("=" * 60)
    
    responses = {}
    
    def timeout_handler(signum, frame):
        raise TimeoutError("User response timeout")
    
    try:
        # Set timeout (Unix only - SIGALRM not available on Windows)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        else:
            print(f"NOTE: Timeout of {timeout_seconds}s disabled (Windows/non-Unix detected).")
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- Question {i}/{len(questions)} ---")
            print(f"\n{question}")
            print("-" * 40)
            
            # Multi-line input support: empty line ends input
            print("(Enter your response, then press Enter twice to submit)")
            lines = []
            while True:
                try:
                    line = input()
                    if line == "" and lines:
                        break
                    lines.append(line)
                except EOFError:
                    break
            
            response = "\n".join(lines).strip()
            if not response:
                response = input("Your response (single line): ").strip()
            
            responses[question] = response
            print(f"✓ Response recorded")
            
            # REMOVED: Duplicate interaction logging. 
            # SupervisorAgent/deciding node is responsible for logging the contextualized interaction.
        
        # Cancel timeout
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        
        print("\n" + "=" * 60)
        print("✓ All responses collected")
        print("=" * 60)
            
    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted by user (Ctrl+C)")
        checkpoint_path = save_checkpoint(state, f"interrupted_{trigger}")
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        print("\nResume later with:")
        print(f"  python -m src.graph --resume {checkpoint_path}")
        raise SystemExit(0)
        
    except TimeoutError:
        print(f"\n\n⚠ User response timeout ({timeout_seconds}s)")
        checkpoint_path = save_checkpoint(state, f"timeout_{trigger}")
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        print("\nResume later with:")
        print(f"  python -m src.graph --resume {checkpoint_path}")
        raise SystemExit(0)
        
    except EOFError:
        print(f"\n\n⚠ End of input (EOF)")
        checkpoint_path = save_checkpoint(state, f"eof_{trigger}")
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        print("\nResume later with:")
        print(f"  python -m src.graph --resume {checkpoint_path}")
        raise SystemExit(0)
    
    # ═══════════════════════════════════════════════════════════════════════
    # VALIDATE USER RESPONSES: Ensure format matches expected trigger type
    # ═══════════════════════════════════════════════════════════════════════
    validation_errors = _validate_user_responses(trigger, responses, questions)
    if validation_errors:
        # Re-prompt with validation errors
        error_msg = "\n".join(f"  - {err}" for err in validation_errors)
        return {
            "pending_user_questions": [
                f"Your response had validation errors:\n{error_msg}\n\n"
                f"Please try again:\n{questions[0] if questions else 'Please provide a valid response.'}"
            ],
            "awaiting_user_input": True,
            "ask_user_trigger": trigger,  # Keep same trigger
            "last_node_before_ask_user": state.get("last_node_before_ask_user"),
        }
    
    return {
        "user_responses": {**state.get("user_responses", {}), **responses},
        "pending_user_questions": [],
        "awaiting_user_input": False,
        "workflow_phase": "awaiting_user",
    }


def material_checkpoint_node(state: ReproState) -> dict:
    """
    Mandatory material validation checkpoint after Stage 0.
    
    This node ALWAYS routes to ask_user to require user confirmation
    of material validation results before proceeding to Stage 1+.
    
    IMPORTANT: This node stores extracted materials in `pending_validated_materials`,
    NOT in `validated_materials`. The supervisor_node will move them to 
    `validated_materials` ONLY when the user approves.
    
    Per global_rules.md RULE 0A:
    "After Stage 0 completes, you MUST pause and ask the user to confirm
    the material optical constants are correct before proceeding."
    
    Returns:
        Dict with state updates including pending_user_questions and pending_validated_materials
    """
    from schemas.state import get_validation_hierarchy
    
    # Get material validation results from progress
    progress = state.get("progress", {})
    stages = progress.get("stages", [])
    
    # Find Stage 0 (material validation) results
    stage0_info = None
    for stage in stages:
        if stage.get("stage_type") == "MATERIAL_VALIDATION":
            stage0_info = stage
            break
    
    # Get output files from stage_outputs
    stage_outputs = state.get("stage_outputs", {})
    output_files = stage_outputs.get("files", [])
    plot_files = [f for f in output_files if f.endswith(('.png', '.pdf', '.jpg'))]
    
    # Extract materials from plan parameters - stored as PENDING until user approves
    pending_materials = _extract_validated_materials(state)
    
    # Guard against empty materials
    warning_msg = ""
    if not pending_materials:
        warning_msg = (
            "\n\n⚠️ WARNING: No materials were automatically detected! "
            "Code generation will FAIL without materials. "
            "Please select 'CHANGE_MATERIAL' or 'CHANGE_DATABASE' to specify them manually."
        )
    
    # Build the checkpoint question per global_rules.md RULE 0A format
    question = _format_material_checkpoint_question(state, stage0_info, plot_files, pending_materials)
    question += warning_msg
    
    return {
        "workflow_phase": "material_checkpoint",
        "pending_user_questions": [question],
        "awaiting_user_input": True,
        "ask_user_trigger": "material_checkpoint",
        "last_node_before_ask_user": "material_checkpoint",
        # Store as PENDING - will be moved to validated_materials on user approval
        "pending_validated_materials": pending_materials,
    }


def _extract_validated_materials(state: ReproState) -> list:
    """
    Extract material information from Stage 0 outputs or fall back to plan data.
    """
    materials = _materials_from_stage_outputs(state)
    if not materials:
        materials.extend(_extract_materials_from_plan_assumptions(state))
    if not materials:
        materials = state.get("planned_materials", [])
    return _deduplicate_materials(materials)


def _load_material_database() -> dict:
    """Load materials/index.json database."""
    import json
    import os
    
    index_path = os.path.join(os.path.dirname(__file__), "..", "materials", "index.json")
    if not os.path.exists(index_path):
        # Try relative to current working directory
        index_path = "materials/index.json"
    
    try:
        with open(index_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load materials/index.json: {e}")
        return {}


def _match_material_from_text(text: str, material_lookup: dict) -> dict:
    """
    Match material name from text against material database.
    
    Returns the best matching material entry or None.
    """
    import re
    text_lower = text.lower()
    
    # Priority 1: Exact material_id match (e.g., "palik_gold")
    for mat_id, mat_entry in material_lookup.items():
        if mat_id in text_lower:
            return mat_entry
    
    # Priority 2: Word-boundary match for simple names
    # This avoids "golden" matching "gold" or "usage" matching "ag"
    simple_names = ["gold", "silver", "aluminum", "silicon", "sio2", "glass", "water", "air", "ag", "au", "al", "si"]
    
    # Map common chemical symbols to full names for lookup
    symbol_map = {
        "ag": "silver",
        "au": "gold",
        "al": "aluminum",
        "si": "silicon"
    }
    
    for name in simple_names:
        # Use regex to match whole words only
        if re.search(r'\b' + re.escape(name) + r'\b', text_lower):
            lookup_name = symbol_map.get(name, name)
            
            # Find the best match in lookup (prefer entries with csv_available=true)
            candidates = [v for k, v in material_lookup.items() if lookup_name in k]
            csv_available = [c for c in candidates if c.get("csv_available", False)]
            
            if csv_available:
                return csv_available[0]
            elif candidates:
                return candidates[0]
    
    return None


def _format_validated_material(mat_entry: dict, from_source: str) -> dict:
    """Format a material database entry for validated_materials list."""
    data_file = mat_entry.get("data_file")
    path = f"materials/{data_file}" if data_file else None
    
    return {
        "material_id": mat_entry.get("material_id"),
        "name": mat_entry.get("name"),
        "source": mat_entry.get("source"),
        "path": path,
        "csv_available": mat_entry.get("csv_available", False),
        "drude_lorentz_fit": mat_entry.get("drude_lorentz_fit"),
        "wavelength_range_nm": mat_entry.get("wavelength_range_nm"),
        "from": from_source,
    }


def _format_material_checkpoint_question(
    state: ReproState, 
    stage0_info: dict, 
    plot_files: list,
    validated_materials: list
) -> str:
    """Format the material checkpoint question per global_rules.md RULE 0A."""
    paper_id = state.get("paper_id", "unknown")
    
    # Format validated materials
    if validated_materials:
        materials_info = []
        for mat in validated_materials:
            mat_name = mat.get('name') or mat.get('material_id', 'unknown')
            materials_info.append(
                f"- {mat_name.upper()}: source={mat.get('source', 'unknown')}, file={mat.get('path', 'N/A')}"
            )
    else:
        materials_info = ["- No materials automatically detected"]
    
    # Format plot files list
    plots_text = "\n".join(f"- {f}" for f in plot_files) if plot_files else "- No plots generated"
    
    question = f"""
═══════════════════════════════════════════════════════════════════════
MANDATORY MATERIAL VALIDATION CHECKPOINT
═══════════════════════════════════════════════════════════════════════

Stage 0 (Material Validation) has completed for paper: {paper_id}

**Validated materials (will be used for all subsequent stages):**
{chr(10).join(materials_info)}

**Generated plots:**
{plots_text}

Please review the material optical constants comparison plots above.

**Required confirmation:**

Do the simulated optical constants (n, k, ε) match the paper's data 
within acceptable tolerance?

Options:
1. APPROVE - Material validation looks correct, proceed to Stage 1
2. CHANGE_DATABASE - Use different material database (specify which)
3. CHANGE_MATERIAL - Paper uses different material than assumed (specify which)
4. NEED_HELP - Unclear how to validate, need guidance

Note: If you APPROVE, the validated_materials list above will be passed
to Code Generator for all subsequent stages.

Please respond with your choice and any notes.
═══════════════════════════════════════════════════════════════════════
"""
    return question


def generate_report_node(state: ReproState) -> ReproState:
    """Generate final reproduction report."""
    state["workflow_phase"] = "reporting"
    
    # Compute token summary
    if "metrics" in state:
        total_input = 0
        total_output = 0
        for call in state["metrics"].get("agent_calls", []):
            # Note: agent calls might not have token counts yet in stub, 
            # but this logic prepares for it.
            total_input += call.get("input_tokens", 0) or 0
            total_output += call.get("output_tokens", 0) or 0
            
        state["metrics"]["token_summary"] = {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "estimated_cost": (total_input * 3.0 + total_output * 15.0) / 1_000_000 # Example pricing
        }
    
    # STUB: Populate report structures if missing
    if "paper_citation" not in state:
        state["paper_citation"] = {
            "title": state.get("paper_title", "Unknown"),
            "authors": "Unknown",
            "journal": "Unknown", 
            "year": 2023
        }
        
    if "executive_summary" not in state:
        state["executive_summary"] = {
            "overall_assessment": [
                {"aspect": "Material Properties", "status": "Reproduced", "status_icon": "✅", "notes": "Validated against Palik"},
                {"aspect": "Geometric Resonances", "status": "Partial", "status_icon": "⚠️", "notes": "Systematic red-shift"}
            ]
        }

    # Build quantitative summary table from analysis_result_reports
    quantitative_reports = state.get("analysis_result_reports", [])
    if quantitative_reports:
        summary_rows: List[Dict[str, Any]] = []
        for report in quantitative_reports:
            metrics = report.get("quantitative_metrics") or {}
            summary_rows.append({
                "stage_id": report.get("stage_id"),
                "figure_id": report.get("target_figure"),
                "status": report.get("status"),
                "precision_requirement": report.get("precision_requirement"),
                "peak_position_error_percent": metrics.get("peak_position_error_percent"),
                "normalized_rmse_percent": metrics.get("normalized_rmse_percent"),
                "correlation": metrics.get("correlation"),
                "n_points_compared": metrics.get("n_points_compared"),
            })
        state["quantitative_summary"] = summary_rows
        
    # TODO: Implement report generation logic
    # - Compile figure comparisons
    # - Document assumptions
    # - Generate REPRODUCTION_REPORT.md
    return state


def handle_backtrack_node(state: ReproState) -> dict:
    """
    Process cross-stage backtracking.
    
    When SupervisorAgent decides to backtrack (verdict="backtrack_to_stage"),
    this node:
    1. Reads the backtrack_decision from state
    2. Marks the target stage as "needs_rerun"
    3. Marks all dependent stages as "invalidated"
    4. Increments the backtrack_count
    5. Clears working data to prepare for re-run
    
    Returns:
        Dict with state updates (LangGraph merges this into state)
    """
    import copy
    
    decision = state.get("backtrack_decision", {})
    if not decision or not decision.get("accepted"):
        # No valid backtrack decision - shouldn't happen, but handle gracefully
        return {"workflow_phase": "backtracking"}
    
    target_id = decision.get("target_stage_id", "")
    stages_to_invalidate = decision.get("stages_to_invalidate", [])
    
    # Deep copy progress to avoid mutating original
    progress = copy.deepcopy(state.get("progress", {}))
    stages = progress.get("stages", [])
    
    # Update stage statuses
    for stage in stages:
        stage_id = stage.get("stage_id", "")
        if stage_id == target_id:
            # Target stage: mark for re-run
            stage["status"] = "needs_rerun"
            # CRITICAL: Clear persisted outputs for the target stage so history matches new state
            stage["outputs"] = []
            stage["discrepancies"] = []
        elif stage_id in stages_to_invalidate:
            # Dependent stages: mark as invalidated
            stage["status"] = "invalidated"
    
    # Build return dict with all state updates
    result = {
        "workflow_phase": "backtracking",
        "progress": progress,
        "current_stage_id": target_id,
        "backtrack_count": state.get("backtrack_count", 0) + 1,
        "backtrack_decision": None, # Clear decision to prevent re-processing
        # Clear working data to prepare for re-run
        "code": None,
        "design_description": None,
        "stage_outputs": {},
        "run_error": None,
        "analysis_summary": None,
        # Track invalidated stages
        "invalidated_stages": stages_to_invalidate,
        # Clear verdicts
        "last_design_review_verdict": None,
        "last_code_review_verdict": None,
        "supervisor_verdict": None,
        # Reset per-stage counters
        "design_revision_count": 0,
        "code_revision_count": 0,
        "execution_failure_count": 0,
        "physics_failure_count": 0,
        "analysis_revision_count": 0,
    }
    
    # Guard clause: Check if max backtracks exceeded
    max_backtracks = state.get("runtime_config", {}).get("max_backtracks", 2)
    if result["backtrack_count"] > max_backtracks:
        # Escalate to user instead of looping forever
        return {
            "workflow_phase": "backtracking_limit",
            "ask_user_trigger": "backtrack_limit", # Needs new trigger type or generic "unknown"
            "pending_user_questions": [
                f"Backtrack limit ({max_backtracks}) exceeded. System is looping. How to proceed?"
            ],
            "awaiting_user_input": True,
            "last_node_before_ask_user": "handle_backtrack"
        }
        
    return result

