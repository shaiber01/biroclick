"""
Analysis agent nodes: results_analyzer_node, comparison_validator_node.

These nodes handle results analysis and comparison validation.

State Keys
----------
results_analyzer_node:
    READS: execution_result, current_stage_id, plan, paper_figures, paper_text,
           digitized_data, stage_outputs, analysis_revision_count
    WRITES: workflow_phase, analysis_reports, stage_outputs, analysis_summary,
            ask_user_trigger, pending_user_questions, awaiting_user_input

comparison_validator_node:
    READS: current_stage_id, analysis_reports, plan, stage_outputs,
           analysis_revision_count, runtime_config
    WRITES: workflow_phase, comparison_verdict, analysis_revision_count,
            stage_comparisons, ask_user_trigger, pending_user_questions,
            awaiting_user_input
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from schemas.state import (
    ReproState,
    get_plan_stage,
    MAX_ANALYSIS_REVISIONS,
)
from src.prompts import build_agent_prompt
from src.llm_client import (
    call_agent_with_metrics,
    build_user_content_for_analyzer,
    get_images_for_analyzer,
)

from .helpers.context import check_context_or_escalate
from .base import with_context_check, increment_counter_with_max
from .helpers.stubs import ensure_stub_figures
from .helpers.numeric import load_numeric_series, quantitative_curve_metrics
from .helpers.validation import (
    classification_from_metrics,
    evaluate_validation_criteria,
    extract_targets_from_feedback,
    match_output_file,
    match_expected_files,
    collect_expected_outputs,
    collect_expected_columns,
    stage_comparisons_for_stage,
    analysis_reports_for_stage,
    validate_analysis_reports,
    breakdown_comparison_classifications,
)
from .helpers.metrics import record_discrepancy
from src.agents.constants import AnalysisClassification

# Project root for path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent


def results_analyzer_node(state: ReproState) -> dict:
    """ResultsAnalyzerAgent: Compare results to paper figures."""
    logger = logging.getLogger(__name__)
    
    context_update = check_context_or_escalate(state, "analyze")
    if context_update:
        if context_update.get("awaiting_user_input"):
            return context_update
        state = {**state, **context_update}
    
    # Validate current_stage_id
    current_stage_id = state.get("current_stage_id")
    if not current_stage_id:
        logger.error(
            "current_stage_id is None - cannot analyze without a selected stage. "
            "This indicates select_stage_node did not run or returned None."
        )
        return {
            "workflow_phase": "analysis",
            "ask_user_trigger": "missing_stage_id",
            "pending_user_questions": [
                "ERROR: No stage selected for analysis. This indicates a workflow error. "
                "Please check stage selection or restart the workflow."
            ],
            "awaiting_user_input": True,
        }
    
    # Build system prompt for analysis
    system_prompt = build_agent_prompt("results_analyzer", state)
    
    stage_info = get_plan_stage(state, current_stage_id) if current_stage_id else None
    figures = ensure_stub_figures(state)
    target_ids: List[str] = []
    # Check if stage explicitly defines targets (empty list means no targets)
    if stage_info:
        if "targets" in stage_info:
            target_ids = stage_info["targets"] if stage_info["targets"] else []
        elif stage_info.get("target_details"):
            target_ids = [t.get("figure_id") for t in stage_info["target_details"] if t.get("figure_id")]
        else:
            # Fall back to figures only if stage doesn't explicitly define targets
            target_ids = [fig.get("id", "FigStub") for fig in figures]
    else:
        target_ids = [fig.get("id", "FigStub") for fig in figures]
    
    # Validate targets exist - check BEFORE file validation
    if not target_ids or len(target_ids) == 0:
        logger.error(
            f"Stage '{current_stage_id}' has no targets defined. "
            "Cannot proceed with analysis without targets."
        )
        return {
            "workflow_phase": "analysis",
            "analysis_summary": {
                "stage_id": current_stage_id,
                "overall_classification": AnalysisClassification.NO_TARGETS,
                "unresolved_targets": [],
                "matched_targets": [],
                "pending_targets": [],
                "missing_targets": [],
                "mismatch_targets": [],
                "summary": "Analysis skipped: Stage has no targets defined.",
                "totals": {
                    "targets": 0,
                    "matches": 0,
                    "pending": 0,
                    "missing": 0,
                    "mismatch": 0,
                },
                "discrepancies_logged": 0,
                "validation_criteria": [],
                "feedback_applied": [],
                "notes": f"Stage {current_stage_id} has no targets - skipping analysis.",
            },
            "analysis_overall_classification": AnalysisClassification.NO_TARGETS,
            "analysis_result_reports": [],
            "figure_comparisons": [],
            "supervisor_verdict": "ok_continue",
            "supervisor_feedback": f"Stage {current_stage_id} has no targets - skipping analysis.",
        }
    
    # Validate output files exist on disk
    paper_id = state.get("paper_id", "unknown")
    base_output_dir = PROJECT_ROOT / "outputs" / paper_id / current_stage_id
    
    # Use stage_outputs.files from state
    stage_outputs = state.get("stage_outputs") or {}
    output_files = stage_outputs.get("files", [])
    
    existing_files: List[str] = []
    missing_files: List[str] = []
    
    if not stage_outputs or not output_files:
        logger.error(
            f"Stage outputs are empty or missing for stage {current_stage_id}. "
            "Cannot proceed with analysis without simulation outputs."
        )
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
    
    # Initialize file lists before the loop (FIX for UnboundLocalError)
    # This was the critical bug: output_files was used in loop but these lists weren't populated if loop didn't run? 
    # No, the lists were initialized AFTER the loop in some versions, or not used correctly.
    # Let's make sure we iterate over the list we just extracted.
    
    for file_path in output_files:
        file_path_str = str(file_path) if not isinstance(file_path, str) else file_path
        file_obj = Path(file_path_str)
        
        if not file_obj.is_absolute():
            file_obj = base_output_dir / file_obj
        
        if not file_obj.exists() and not Path(file_path_str).is_absolute():
            file_obj = PROJECT_ROOT / file_path_str
        
        if file_obj.exists() and file_obj.is_file():
            existing_files.append(str(file_obj.resolve()))
        else:
            missing_files.append(file_path_str)
    
    if not existing_files and output_files:
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
    
    if missing_files:
        logger.warning(
            f"Some output files are missing: {missing_files}. "
            f"Proceeding with {len(existing_files)} available files."
        )
        output_files = existing_files
    
    expected_outputs_map = collect_expected_outputs(stage_info, paper_id, current_stage_id or "")
    plan_stage_columns = collect_expected_columns(stage_info)
    figure_lookup = {fig.get("id") or fig.get("figure_id"): fig for fig in figures}
    if stage_info and stage_info.get("target_details"):
        target_meta_list = stage_info["target_details"]
    else:
        target_meta_list = (state.get("plan") or {}).get("targets", [])
    plan_targets_map = {target.get("figure_id"): target for target in target_meta_list}
    
    matched_targets: List[str] = []
    pending_targets: List[str] = []
    missing_targets: List[str] = []
    mismatch_targets: List[str] = []
    stage_discrepancies: List[Dict[str, Any]] = []
    figure_comparisons: List[Dict[str, Any]] = []
    per_result_reports: List[Dict[str, Any]] = []
    
    # Prioritize feedback targets, then others
    feedback_targets = extract_targets_from_feedback(state.get("analysis_feedback", ""), target_ids)
    ordered_targets = feedback_targets + [t for t in target_ids if t not in feedback_targets]
    ordered_targets = ordered_targets or target_ids
    
    for target_id in ordered_targets:
        expected_names = expected_outputs_map.get(target_id, [])
        matched_file = match_expected_files(expected_names, output_files)
        if not matched_file:
            matched_file = match_output_file(output_files, target_id)
        has_output = matched_file is not None
        figure_meta = figure_lookup.get(target_id) or {}
        target_cfg = plan_targets_map.get(target_id, {})
        precision_requirement = target_cfg.get("precision_requirement", "acceptable")
        
        reference_path = target_cfg.get("reference_data_path")
        if not reference_path and stage_info:
            reference_path = stage_info.get("reference_data_path")
        digitized_path = (
            figure_meta.get("digitized_data_path")
            or target_cfg.get("digitized_data_path")
            or target_cfg.get("digitized_reference")
            or reference_path
        )
        
        paper_image_path = figure_meta.get("image_path")
        if paper_image_path:
            image_file = Path(paper_image_path)
            if not image_file.exists() or not image_file.is_file():
                logger.warning(
                    f"Figure image path does not exist or is not a file: {paper_image_path}. "
                    f"Comparison for {target_id} will proceed without reference image."
                )
                paper_image_path = None  # Set to None when file doesn't exist
        
        requires_digitized = precision_requirement == "excellent"
        quantitative_metrics: Dict[str, Any] = {}
        classification_label = "missing_output"
        
        # Enforce digitized data requirement
        if requires_digitized and not digitized_path:
            logger.error(
                f"Target '{target_id}' requires 'excellent' precision (<2%) but no digitized data path provided. "
                "Cannot perform quantitative comparison without digitized reference data."
            )
            pending_targets.append(target_id)
            stage_discrepancies.append(
                record_discrepancy(
                    state,
                    current_stage_id,
                    target_id,
                    "digitized_data",
                    "Digitized reference data required for excellent precision",
                    "Not provided",
                    classification="investigate",
                    likely_cause="Target requires <2% precision but digitized data path is missing.",
                    action_taken="Analysis blocked until digitized data is provided",
                    blocking=True,
                )
            )
            classification_label = "missing_digitized_data"
            figure_comparisons.append({
                "figure_id": target_id,
                "target_id": target_id,  # Keep for backward compatibility
                "stage_id": current_stage_id,
                "status": "missing_digitized_data",
                "classification": "missing_digitized_data",
                "quantitative_metrics": {},
                "notes": "Analysis blocked: Digitized data required but not provided",
                "title": figure_meta.get("description", target_id),
                "paper_image_path": paper_image_path,
                "reproduction_image_path": None,
                "comparison_table": [],
                "shape_comparison": [],
                "reason_for_difference": "Digitized reference data required but not provided",
            })
            # Add report entry even when digitized data is missing
            expected_names = expected_outputs_map.get(target_id, [])
            per_result_reports.append({
                "result_id": f"{current_stage_id or 'stage'}_{target_id}",
                "target_figure": target_id,
                "status": classification_label,
                "expected_outputs": expected_names,
                "matched_output": matched_file if has_output else None,
                "precision_requirement": precision_requirement,
                "digitized_data_path": None,
                "validation_criteria": [],
                "quantitative_metrics": {},
                "criteria_failures": ["Digitized reference data required but not provided"],
                "notes": "Analysis blocked: Digitized data required but not provided",
            })
            continue
        
        if not has_output:
            missing_targets.append(target_id)
            stage_discrepancies.append(
                record_discrepancy(
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
            sim_series = load_numeric_series(matched_file, expected_columns)
            ref_series = load_numeric_series(digitized_path, expected_columns)
            quantitative_metrics = quantitative_curve_metrics(sim_series, ref_series)
            has_reference = ref_series is not None
            
            classification_label = classification_from_metrics(
                quantitative_metrics,
                precision_requirement,
                has_reference,
            )
            if classification_label == AnalysisClassification.MATCH:
                matched_targets.append(target_id)
            elif classification_label in {AnalysisClassification.PENDING_VALIDATION, AnalysisClassification.PARTIAL_MATCH}:
                pending_targets.append(target_id)
            else:
                mismatch_targets.append(target_id)
            
            error_percent = quantitative_metrics.get("peak_position_error_percent")
            if error_percent is not None and classification_label in {AnalysisClassification.PARTIAL_MATCH, AnalysisClassification.MISMATCH}:
                discrepancy_class = "acceptable" if classification_label == AnalysisClassification.PARTIAL_MATCH else "investigate"
                blocking = classification_label == AnalysisClassification.MISMATCH
                paper_peak = quantitative_metrics.get("peak_position_paper")
                sim_peak = quantitative_metrics.get("peak_position_sim")
                stage_discrepancies.append(
                    record_discrepancy(
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
                "✅ Match" if has_output and classification_label == AnalysisClassification.MATCH
                else "⚠️ Pending" if classification_label in {AnalysisClassification.PENDING_VALIDATION, AnalysisClassification.PARTIAL_MATCH}
                else "❌ Missing"
            ),
        }]
        if digitized_path:
            comparison_table.append({
                "feature": "Digitized reference",
                "paper": digitized_path,
                "reproduction": matched_file or "Not generated",
                "status": (
                    "Pending review" if classification_label not in {"missing_output", AnalysisClassification.MISMATCH}
                    else "❌ Missing"
                ),
            })
        
        comparison_entry = {
            "figure_id": target_id,
            "stage_id": current_stage_id,
            "title": figure_meta.get("description", target_id),
            "paper_image_path": paper_image_path,
            "reproduction_image_path": matched_file,
            "comparison_table": comparison_table,
            "shape_comparison": [],
            "reason_for_difference": "" if has_output else "No simulation output matched this figure.",
            "classification": classification_label,
        }
        figure_comparisons.append(comparison_entry)
        
        target_criteria: List[str] = []
        if stage_info:
            for criterion in stage_info.get("validation_criteria", []):
                if target_id and target_id.lower() in criterion.lower():
                    target_criteria.append(criterion)
        criteria_passed, criteria_failures = evaluate_validation_criteria(quantitative_metrics, target_criteria)
        if target_criteria and not criteria_failures and not quantitative_metrics:
            criteria_failures.append("Validation criteria defined but quantitative metrics missing.")
        if criteria_failures:
            if target_id not in mismatch_targets:
                mismatch_targets.append(target_id)
            classification_label = AnalysisClassification.MISMATCH
            for failure in criteria_failures:
                sim_value_display = json.dumps(quantitative_metrics) if quantitative_metrics else "N/A"
                stage_discrepancies.append(
                    record_discrepancy(
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
    
    # Check if any reports have missing/incomplete metrics
    has_incomplete_metrics = False
    for report in per_result_reports:
        metrics = report.get("quantitative_metrics", {})
        # Check if critical metrics are missing (None or empty)
        if not metrics or metrics.get("peak_position_error_percent") is None:
            # If precision requirement is excellent or acceptable, missing metrics is a problem
            precision = report.get("precision_requirement", "qualitative")
            if precision in ("excellent", "acceptable"):
                has_incomplete_metrics = True
                break
    
    if total_targets == 0:
        overall_classification = AnalysisClassification.NO_TARGETS
    elif missing_count > 0 or mismatch_count > 0:
        overall_classification = AnalysisClassification.POOR_MATCH
    elif pending_count > 0 or has_incomplete_metrics:
        overall_classification = AnalysisClassification.PARTIAL_MATCH
    elif len(matched_targets) == total_targets:
        overall_classification = AnalysisClassification.EXCELLENT_MATCH
    else:
        overall_classification = AnalysisClassification.ACCEPTABLE_MATCH
    
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
    
    # Multimodal LLM call for visual comparison
    images = get_images_for_analyzer(state)
    
    if images and figure_comparisons:
        user_content = build_user_content_for_analyzer(state)
        user_content += f"\n\n# QUANTITATIVE ANALYSIS RESULTS\n\nOverall: {overall_classification}\n"
        user_content += f"Matched: {len(matched_targets)}, Pending: {pending_count}, Missing: {missing_count}, Mismatch: {mismatch_count}"
        
        try:
            llm_analysis = call_agent_with_metrics(
                agent_name="results_analyzer",
                system_prompt=system_prompt,
                user_content=user_content,
                state=state,
                images=images[:10],
            )
            
            if llm_analysis:
                if llm_analysis.get("overall_classification"):
                    overall_classification = llm_analysis["overall_classification"]
                
                if llm_analysis.get("summary"):
                    summary["llm_qualitative_analysis"] = llm_analysis["summary"]
                
                for llm_comp in llm_analysis.get("figure_comparisons", []):
                    fig_id = llm_comp.get("figure_id")
                    for existing_comp in figure_comparisons:
                        if existing_comp.get("figure_id") == fig_id:
                            existing_comp["shape_comparison"] = llm_comp.get("shape_comparison", [])
                            existing_comp["reason_for_difference"] = llm_comp.get("reason_for_difference", "")
                            break
                
        except Exception as e:
            logger.warning(f"Visual analysis LLM call failed: {e}. Using quantitative results only.")
    
    return {
        "workflow_phase": "analysis",
        "analysis_summary": summary,
        "analysis_overall_classification": overall_classification,
        "analysis_result_reports": merged_reports,
        "figure_comparisons": filtered_existing + figure_comparisons,
        "analysis_feedback": analysis_feedback_next,
    }


@with_context_check("comparison_check")
def comparison_validator_node(state: ReproState) -> dict:
    """
    ComparisonValidatorAgent: Validate comparison accuracy.
    
    Returns dict with state updates (LangGraph merges this into state).
    
    IMPORTANT: 
    - Sets `comparison_verdict` state field from agent output's `verdict`.
    - Increments `analysis_revision_count` when verdict is "needs_revision".
    
    Note: Context check is handled by @with_context_check decorator.
    """

    # Defensive check: if state already has awaiting_user_input=True from a previous node,
    # skip processing and return immediately. The @with_context_check decorator handles
    # escalations from this node, but this guards against pre-existing escalation state.
    if state.get("awaiting_user_input"):
        return state
    
    stage_id = state.get("current_stage_id")
    comparisons = stage_comparisons_for_stage(state, stage_id)
    breakdown = breakdown_comparison_classifications(comparisons)
    stage_info = get_plan_stage(state, stage_id) if stage_id else None
    if stage_info:
        expected_targets = stage_info.get("targets") or [
            t.get("figure_id") for t in stage_info.get("target_details", []) if t.get("figure_id")
        ]
    else:
        expected_targets = []
    analysis_reports = analysis_reports_for_stage(state, stage_id)
    report_issues = validate_analysis_reports(analysis_reports)
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
    
    # Combine all issues: missing comparisons, report issues, and missing reports
    all_issues = []
    if missing_comparisons:
        all_issues.append(f"Results analyzer did not produce comparisons for: {', '.join(missing_comparisons)}")
    if report_issues:
        all_issues.extend(report_issues)
    if missing_report_targets:
        all_issues.append(f"Missing quantitative reports for: {', '.join(missing_report_targets)}")
    
    if all_issues:
        verdict = "needs_revision"
        feedback = "; ".join(all_issues[:3])
        if len(all_issues) > 3:
            feedback += f" (+{len(all_issues)-3} more)"
    
    result: Dict[str, Any] = {
        "workflow_phase": "comparison_validation",
        "comparison_verdict": verdict,
        "comparison_feedback": feedback,
    }
    
    if verdict == "needs_revision":
        new_count, _ = increment_counter_with_max(
            state, "analysis_revision_count", "max_analysis_revisions", MAX_ANALYSIS_REVISIONS
        )
        result["analysis_revision_count"] = new_count
        result["analysis_feedback"] = feedback
    else:
        result["analysis_feedback"] = None
    
    return result
