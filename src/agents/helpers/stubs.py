"""
Stub builders for generating placeholder plans and stages.

These are used when LLM calls fail or for testing purposes.
"""

from typing import Dict, Any, List

from schemas.state import ReproState


def ensure_stub_figures(state: ReproState) -> List[Dict[str, Any]]:
    """Return available paper figures or a placeholder stub."""
    figures = state.get("paper_figures") or []
    if figures:
        return figures
    return [{
        "id": "FigStub",
        "description": "Placeholder figure generated for stub planning",
        "image_path": "",
    }]


def build_stub_targets(figures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def build_stub_expected_outputs(paper_id: str, stage_id: str, target_ids: List[str], columns: List[str]) -> List[Dict[str, Any]]:
    """Build expected output specifications for a stage."""
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


def build_stub_stages(paper_id: str, targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build default Stage 0 and Stage 1 structures."""
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
        "expected_outputs": build_stub_expected_outputs(
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
        "expected_outputs": build_stub_expected_outputs(
            paper_id,
            "stage1_primary_structure",
            stage1_targets,
            ["wavelength_nm", "transmission"]
        ),
        "reference_data_path": None,
    }
    
    return [stage0, stage1]


def build_stub_planned_materials(state: ReproState) -> List[Dict[str, Any]]:
    """Build placeholder material entries."""
    domain = state.get("paper_domain", "generic")
    return [{
        "material_id": f"{domain}_placeholder",
        "name": f"{domain.title()} Material",
        "source": "stub",
        "path": "materials/placeholder.csv",
    }]


def build_stub_assumptions() -> Dict[str, Any]:
    """Build empty assumptions structure."""
    return {
        "global_assumptions": {
            "materials": [],
            "geometry": [],
            "sources": [],
        },
        "stage_specific": [],
    }


def build_stub_plan(state: ReproState) -> Dict[str, Any]:
    """Build a complete stub plan from state."""
    paper_id = state.get("paper_id", "paper_stub")
    paper_title = state.get("paper_title", paper_id.replace("_", " ").title())
    figures = ensure_stub_figures(state)
    targets = build_stub_targets(figures)
    stages = build_stub_stages(paper_id, targets)
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

