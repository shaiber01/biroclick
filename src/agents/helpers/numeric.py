"""
Numeric and curve analysis utilities.

Functions for loading data files, normalizing series, and computing
quantitative metrics for curve comparison.
"""

import csv
import json
from math import isfinite
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np

# PROJECT_ROOT is set at module level for path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def resolve_data_path(path_str: Optional[str]) -> Optional[Path]:
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


def normalize_series(xs: List[float], ys: List[float]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Normalize and sort x,y series data.
    
    Args:
        xs: List of x values
        ys: List of y values
        
    Returns:
        Tuple of sorted numpy arrays (x, y), or None if invalid
    """
    if not xs or not ys or len(xs) != len(ys):
        return None
    pairs = sorted(zip(xs, ys), key=lambda item: item[0])
    x_arr = np.array([float(p[0]) for p in pairs], dtype=float)
    y_arr = np.array([float(p[1]) for p in pairs], dtype=float)
    if len(x_arr) < 3:
        return None
    return x_arr, y_arr


def units_to_multiplier(header: str) -> float:
    """Convert column header units to nm multiplier."""
    header_lower = header.lower()
    if header_lower.endswith("_nm") or "(nm)" in header_lower:
        return 1.0
    if header_lower.endswith("_um") or "(um)" in header_lower:
        return 1000.0
    if header_lower.endswith("_m") or "(m)" in header_lower:
        return 1e9
    return 1.0


def load_numeric_series(path_str: Optional[str], columns: Optional[List[str]] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load x,y series data from various file formats.
    
    Supports: CSV, TSV, JSON, NPY, NPZ
    
    Args:
        path_str: Path to the data file
        columns: Optional column names to look for [x_col, y_col]
        
    Returns:
        Tuple of numpy arrays (x, y), or None if loading fails
    """
    resolved = resolve_data_path(path_str)
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
                                x_multiplier = units_to_multiplier(field_lower)
                            if col_y and col_y.lower() in field_lower and y_idx is None:
                                y_idx = idx
                                y_multiplier = units_to_multiplier(field_lower)
                    if x_idx is None:
                        x_idx = next((i for i, field in enumerate(normalized_header) if "wave" in field.lower()), None)
                        if x_idx is not None:
                            x_multiplier = units_to_multiplier(normalized_header[x_idx])
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
            return normalize_series(xs, ys)
        if suffix == ".json":
            data = json.loads(resolved.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "x" in data and "y" in data:
                xs = [float(x) for x in data["x"]]
                ys = [float(y) for y in data["y"]]
                return normalize_series(xs, ys)
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
                return normalize_series(xs, ys)
            return None
        if suffix == ".npy":
            arr = np.load(resolved)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return normalize_series(arr[:, 0].tolist(), arr[:, 1].tolist())
            return None
        if suffix == ".npz":
            archive = np.load(resolved)
            first_key = archive.files[0] if archive.files else None
            if not first_key:
                return None
            arr = archive[first_key]
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return normalize_series(arr[:, 0].tolist(), arr[:, 1].tolist())
            return None
    except Exception:
        return None
    return None


def compute_peak_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Compute basic peak metrics for a curve.
    
    Args:
        x: Wavelength/frequency array
        y: Intensity/response array
        
    Returns:
        Dict with peak_position, peak_value, fwhm (if computable)
    """
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


def quantitative_curve_metrics(
    sim_series: Optional[Tuple[np.ndarray, np.ndarray]],
    ref_series: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    """
    Compute quantitative metrics comparing simulation vs reference curves.
    
    Args:
        sim_series: Simulation data as (x, y) arrays
        ref_series: Reference data as (x, y) arrays
        
    Returns:
        Dict with various comparison metrics
    """
    metrics: Dict[str, float] = {}
    if not sim_series:
        return metrics
    x_sim, y_sim = sim_series
    sim_peak = compute_peak_metrics(x_sim, y_sim)
    metrics["peak_position_sim"] = sim_peak.get("peak_position")
    metrics["peak_value_sim"] = sim_peak.get("peak_value")
    metrics["fwhm_sim"] = sim_peak.get("fwhm")
    
    if not ref_series:
        return metrics
    
    x_ref, y_ref = ref_series
    if x_ref.size < 3 or y_ref.size < 3:
        return metrics
    ref_peak = compute_peak_metrics(x_ref, y_ref)
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



