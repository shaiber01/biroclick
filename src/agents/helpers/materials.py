"""
Material extraction and validation utilities.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from schemas.state import ReproState, get_progress_stage
from .validation import normalize_output_file_entry


def materials_from_stage_outputs(state: ReproState) -> List[Dict[str, Any]]:
    """Build material entries from Stage 0 artifacts (live or archived)."""
    def _build_from_entries(entries: List[Any], source_label: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for entry in entries:
            path_str = normalize_output_file_entry(entry)
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


def load_material_database() -> dict:
    """Load materials/index.json database."""
    index_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "materials", "index.json")
    if not os.path.exists(index_path):
        # Try relative to current working directory
        index_path = "materials/index.json"
    
    try:
        with open(index_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError, OSError) as e:
        print(f"Warning: Could not load materials/index.json: {e}")
        return {}


def get_material_summary_for_prompt() -> str:
    """Generate a compact material summary for LLM prompts.
    
    This is used to provide SimulationDesigner with available material_ids
    so it can reference them correctly instead of inventing file paths.
    """
    db = load_material_database()
    if not db:
        return "## Available Materials\n\nNo material database found."
    
    lines = [
        "## Available Materials (from materials/index.json)",
        "Use these exact `material_id` values in your design. Do NOT invent file paths.",
        "",
        "| material_id | name | csv_available |",
        "|-------------|------|---------------|"
    ]
    for mat in db.get("materials", []):
        mat_id = mat.get("material_id", "")
        name = mat.get("name", "")
        csv = "Yes" if mat.get("csv_available") else "No"
        lines.append(f"| {mat_id} | {name} | {csv} |")
    
    return "\n".join(lines)


def match_material_from_text(text: str, material_lookup: dict) -> Optional[dict]:
    """
    Match material name from text against material database.
    
    Returns the best matching material entry or None.
    """
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


def format_validated_material(mat_entry: dict, from_source: str) -> dict:
    """Format a material database entry for validated_materials list."""
    data_file = mat_entry.get("data_file")
    # Handle empty string as None
    path = f"materials/{data_file}" if data_file and data_file.strip() else None
    
    # Handle None explicitly - if csv_available is None, default to False
    csv_available = mat_entry.get("csv_available")
    if csv_available is None:
        csv_available = False
    
    return {
        "material_id": mat_entry.get("material_id"),
        "name": mat_entry.get("name"),
        "source": mat_entry.get("source"),
        "path": path,
        "csv_available": csv_available,
        "drude_lorentz_fit": mat_entry.get("drude_lorentz_fit"),
        "wavelength_range_nm": mat_entry.get("wavelength_range_nm"),
        "from": from_source,
    }


def extract_materials_from_plan_assumptions(state: ReproState) -> List[Dict[str, Any]]:
    """Existing fallback path that scans plan parameters and assumptions."""
    plan = state.get("plan", {})
    extracted_params = plan.get("extracted_parameters", [])
    assumptions = state.get("assumptions", {})
    
    # Load material database
    material_db = load_material_database()
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
            matched_material = match_material_from_text(value, material_lookup)
            if matched_material and matched_material["material_id"] not in seen_material_ids:
                validated_materials.append(format_validated_material(
                    matched_material, 
                    from_source=f"parameter: {param.get('name')}"
                ))
                seen_material_ids.add(matched_material["material_id"])
    
    global_assumptions = assumptions.get("global_assumptions", {})
    material_assumptions = global_assumptions.get("materials", [])
    
    for assumption in material_assumptions:
        if isinstance(assumption, dict):
            desc = assumption.get("description", "").lower()
            matched_material = match_material_from_text(desc, material_lookup)
            if matched_material and matched_material["material_id"] not in seen_material_ids:
                validated_materials.append(format_validated_material(
                    matched_material,
                    from_source=f"assumption: {assumption.get('description', '')[:50]}"
                ))
                seen_material_ids.add(matched_material["material_id"])
    
    return validated_materials


def deduplicate_materials(materials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate materials based on path or material_id."""
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for mat in materials:
        key = mat.get("path") or mat.get("material_id")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(mat)
    return deduped


def extract_validated_materials(state: ReproState) -> list:
    """
    Extract material information from Stage 0 outputs or fall back to plan data.
    """
    materials = materials_from_stage_outputs(state)
    if not materials:
        materials.extend(extract_materials_from_plan_assumptions(state))
    if not materials:
        materials = state.get("planned_materials", [])
    return deduplicate_materials(materials)


def format_material_checkpoint_question(
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



