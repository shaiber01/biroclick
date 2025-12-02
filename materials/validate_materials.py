#!/usr/bin/env python3
"""
Material Database Validation Script

Validates optical constant data for physical consistency and data quality.
Checks include:
- Kramers-Kronig consistency
- Physical bounds (n>0, k>=0)
- Data continuity (no suspicious jumps)
- Wavelength coverage
- Comparison between tabulated data and Drude-Lorentz fits

Usage:
    python validate_materials.py                    # Validate all materials
    python validate_materials.py palik_gold         # Validate specific material
    python validate_materials.py --plot             # Generate comparison plots
    python validate_materials.py --report           # Generate JSON report
"""

import numpy as np
import json
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import warnings

# Physical constants
HBAR_EV_S = 6.582119569e-16  # hbar in eV·s
C_NM_S = 2.99792458e17      # speed of light in nm/s
HC_EV_NM = 1239.84193       # hc in eV·nm


def wavelength_to_energy_eV(wavelength_nm: np.ndarray) -> np.ndarray:
    """Convert wavelength (nm) to photon energy (eV)."""
    return HC_EV_NM / wavelength_nm


def energy_to_wavelength_nm(energy_eV: np.ndarray) -> np.ndarray:
    """Convert photon energy (eV) to wavelength (nm)."""
    return HC_EV_NM / energy_eV


def load_material_csv(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load wavelength, n, k from CSV file.
    
    Returns:
        wavelength_nm, n, k arrays
    """
    # Count header lines (starting with #)
    with open(filepath, 'r') as f:
        skip_rows = 0
        for line in f:
            if line.strip().startswith('#') or line.strip() == '':
                skip_rows += 1
            elif line.strip().startswith('wavelength'):
                skip_rows += 1
                break
            else:
                break
    
    data = np.loadtxt(filepath, delimiter=',', skiprows=skip_rows, unpack=True)
    return data[0], data[1], data[2]


def compute_epsilon_from_nk(n: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute complex permittivity from n, k.
    
    ε = (n + ik)² = n² - k² + 2ink
    ε₁ = n² - k²
    ε₂ = 2nk
    """
    eps1 = n**2 - k**2
    eps2 = 2 * n * k
    return eps1, eps2


def compute_drude_lorentz_epsilon(
    energy_eV: np.ndarray,
    eps_inf: float,
    drude_terms: List[Dict],
    lorentz_terms: List[Dict]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute permittivity from Drude-Lorentz model.
    
    ε(ω) = ε_∞ - Σ (ωp²)/(ω² + iγω)  [Drude]
             + Σ (σ·ω₀²)/(ω₀² - ω² - iγω)  [Lorentz]
    
    Returns:
        eps1, eps2 (real and imaginary parts)
    """
    omega = energy_eV  # Using eV as frequency unit
    eps = np.ones_like(omega, dtype=complex) * eps_inf
    
    # Drude terms (free electron response)
    for drude in drude_terms:
        omega_p = drude['omega_p_eV']
        gamma = drude['gamma_eV']
        eps -= omega_p**2 / (omega**2 + 1j * gamma * omega)
    
    # Lorentz terms (interband transitions)
    for lorentz in lorentz_terms:
        omega_0 = lorentz['omega_0_eV']
        gamma = lorentz['gamma_eV']
        sigma = lorentz['sigma']
        # Standard Lorentz oscillator formula
        # Note: gamma provides damping that prevents true singularity
        denom = omega_0**2 - omega**2 - 1j * gamma * omega
        eps += sigma * omega_0**2 / denom
    
    return eps.real, eps.imag


def check_physical_bounds(
    wavelength_nm: np.ndarray, 
    n: np.ndarray, 
    k: np.ndarray,
    material_type: str = 'unknown'
) -> Tuple[bool, List[str]]:
    """
    Check that n, k values are physically reasonable.
    
    Rules:
    - n must be positive (always)
    - k must be non-negative (always)
    - For dielectrics: k should be very small in transparent region
    - For metals: n < 1 expected in plasmonic region
    """
    issues = []
    
    # n should be positive
    neg_n_idx = np.where(n <= 0)[0]
    if len(neg_n_idx) > 0:
        issues.append(f"ERROR: n ≤ 0 at {len(neg_n_idx)} wavelengths "
                     f"(first at {wavelength_nm[neg_n_idx[0]]:.1f}nm)")
    
    # k should be non-negative
    neg_k_idx = np.where(k < -1e-10)[0]  # Small tolerance for numerical noise
    if len(neg_k_idx) > 0:
        issues.append(f"ERROR: k < 0 at {len(neg_k_idx)} wavelengths "
                     f"(first at {wavelength_nm[neg_k_idx[0]]:.1f}nm)")
    
    # Type-specific checks
    if material_type == 'dielectric':
        # Dielectrics should have small k in most of the range
        high_k_idx = np.where(k > 0.1)[0]
        if len(high_k_idx) > len(wavelength_nm) * 0.5:
            issues.append(f"WARNING: Dielectric has high absorption (k>0.1) "
                         f"at {len(high_k_idx)} wavelengths")
    
    elif material_type == 'metal':
        # Noble metals (Ag, Au, Cu, Al) typically have n < 1 in plasmonic region
        # But transition metals (Cr, Ti, Ni, Pt) can have n > 1 throughout visible
        # Only flag if we expected plasmonic behavior
        pass  # Removed overly strict check - n>1 is valid for many metals
    
    return len([i for i in issues if 'ERROR' in i]) == 0, issues


def check_continuity(
    wavelength_nm: np.ndarray, 
    n: np.ndarray, 
    k: np.ndarray,
    n_threshold: float = 0.5,
    k_threshold: float = 0.5
) -> Tuple[bool, List[str]]:
    """
    Check for discontinuities in data that might indicate errors.
    
    Large jumps between adjacent points suggest data entry errors
    or incorrect stitching of datasets.
    """
    issues = []
    
    # Compute relative changes
    dn = np.abs(np.diff(n))
    dk = np.abs(np.diff(k))
    
    # Normalize by local value to catch relative jumps
    n_mid = (n[:-1] + n[1:]) / 2
    k_mid = (k[:-1] + k[1:]) / 2
    
    # Large absolute jumps
    n_jumps = np.where(dn > n_threshold)[0]
    k_jumps = np.where(dk > k_threshold)[0]
    
    for idx in n_jumps[:3]:  # Report first 3 only
        issues.append(f"WARNING: Large n discontinuity at {wavelength_nm[idx]:.0f}nm "
                     f"(Δn = {dn[idx]:.3f})")
    
    for idx in k_jumps[:3]:
        issues.append(f"WARNING: Large k discontinuity at {wavelength_nm[idx]:.0f}nm "
                     f"(Δk = {dk[idx]:.3f})")
    
    if len(n_jumps) > 3:
        issues.append(f"... and {len(n_jumps) - 3} more n discontinuities")
    if len(k_jumps) > 3:
        issues.append(f"... and {len(k_jumps) - 3} more k discontinuities")
    
    return len(n_jumps) == 0 and len(k_jumps) == 0, issues


def check_kramers_kronig_consistency(
    wavelength_nm: np.ndarray,
    n: np.ndarray, 
    k: np.ndarray,
    tolerance: float = 0.3
) -> Tuple[bool, List[str]]:
    """
    Check Kramers-Kronig consistency (simplified check).
    
    The KK relations connect ε₁ and ε₂:
    ε₁(ω) = 1 + (2/π) P∫ [ω'ε₂(ω')/(ω'² - ω²)] dω'
    
    A simplified check: near absorption peaks (high k), 
    the refractive index n should show anomalous dispersion.
    """
    issues = []
    
    # Convert to energy for better physical interpretation
    energy_eV = wavelength_to_energy_eV(wavelength_nm)
    
    # Find absorption peaks
    k_mean = np.mean(k)
    k_std = np.std(k)
    
    if k_std < 1e-6:
        # Uniform k - likely a transparent dielectric
        return True, []
    
    # Compute derivative of n with respect to energy
    dn_dE = np.gradient(n, energy_eV)
    
    # Near strong absorption (high k), check for dispersion features
    high_k_mask = k > k_mean + 0.5 * k_std
    high_k_regions = np.where(high_k_mask)[0]
    
    if len(high_k_regions) > 0:
        # Check that n has some structure (not completely flat)
        n_in_absorption = n[high_k_mask]
        if np.std(n_in_absorption) < 0.01 * np.mean(n_in_absorption):
            issues.append("WARNING: n appears flat in absorption region - "
                         "possible KK inconsistency")
    
    return len(issues) == 0, issues


def check_fit_accuracy(
    wavelength_nm: np.ndarray,
    n: np.ndarray,
    k: np.ndarray,
    fit_params: Dict,
    fit_range: Optional[List[float]] = None
) -> Tuple[bool, List[str], Dict]:
    """
    Compare tabulated data to Drude-Lorentz fit.
    
    Returns fit quality metrics.
    """
    issues = []
    metrics = {}
    
    if not fit_params:
        return True, ["INFO: No Drude-Lorentz fit available"], {}
    
    eps_inf = fit_params.get('eps_inf', 1.0)
    drude_terms = fit_params.get('drude_terms', [])
    lorentz_terms = fit_params.get('lorentz_terms', [])
    
    # Get fit range
    if fit_range is None:
        fit_range = fit_params.get('fit_wavelength_range_nm', [wavelength_nm.min(), wavelength_nm.max()])
    
    # Filter to fit range
    mask = (wavelength_nm >= fit_range[0]) & (wavelength_nm <= fit_range[1])
    if np.sum(mask) < 5:
        return True, ["INFO: Not enough points in fit range"], {}
    
    wl_fit = wavelength_nm[mask]
    n_fit = n[mask]
    k_fit = k[mask]
    
    # Compute permittivity from tabulated data
    eps1_tab, eps2_tab = compute_epsilon_from_nk(n_fit, k_fit)
    
    # Compute permittivity from fit
    energy_eV = wavelength_to_energy_eV(wl_fit)
    eps1_fit, eps2_fit = compute_drude_lorentz_epsilon(
        energy_eV, eps_inf, drude_terms, lorentz_terms
    )
    
    # Compute errors
    eps1_error = np.abs(eps1_fit - eps1_tab)
    eps2_error = np.abs(eps2_fit - eps2_tab)
    
    # Use normalized error to avoid division-by-zero issues
    # Normalize by max value in the dataset instead of point-by-point
    eps1_scale = max(np.abs(eps1_tab).max(), 1.0)
    eps2_scale = max(np.abs(eps2_tab).max(), 1.0)
    
    eps1_norm_error = eps1_error / eps1_scale
    eps2_norm_error = eps2_error / eps2_scale
    
    # Cap individual relative errors to avoid inf values
    eps1_rel_error = np.minimum(eps1_error / (np.abs(eps1_tab) + 0.1 * eps1_scale), 10.0)
    eps2_rel_error = np.minimum(eps2_error / (np.abs(eps2_tab) + 0.1 * eps2_scale), 10.0)
    
    # Metrics - use normalized error for overall assessment
    metrics['eps1_mean_rel_error'] = float(np.mean(eps1_rel_error))
    metrics['eps2_mean_rel_error'] = float(np.mean(eps2_rel_error))
    metrics['eps1_max_rel_error'] = float(np.min([np.max(eps1_rel_error), 10.0]))
    metrics['eps2_max_rel_error'] = float(np.min([np.max(eps2_rel_error), 10.0]))
    
    # Quality assessment based on normalized mean error
    mean_error = (np.mean(eps1_norm_error) + np.mean(eps2_norm_error)) / 2
    
    if mean_error < 0.05:
        quality = 'excellent'
    elif mean_error < 0.15:
        quality = 'good'
    elif mean_error < 0.30:
        quality = 'moderate'
    else:
        quality = 'poor'
    
    metrics['overall_quality'] = quality
    
    # Check against claimed quality
    claimed_quality = fit_params.get('fit_quality', 'unknown')
    if claimed_quality != 'unknown':
        quality_order = {'excellent': 0, 'good': 1, 'moderate': 2, 'approximate': 3, 'poor': 4}
        if quality_order.get(quality, 5) > quality_order.get(claimed_quality, 5) + 1:
            issues.append(f"WARNING: Fit quality ({quality}) worse than claimed ({claimed_quality})")
    
    if mean_error > 0.30:
        issues.append(f"WARNING: Poor fit accuracy (mean relative error: {mean_error:.1%})")
    
    return len([i for i in issues if 'ERROR' in i]) == 0, issues, metrics


def validate_material(
    material: Dict,
    materials_dir: Path,
    verbose: bool = True
) -> Dict:
    """
    Validate a single material.
    
    Returns validation results dictionary.
    """
    mat_id = material['material_id']
    results = {
        'material_id': mat_id,
        'name': material.get('name', mat_id),
        'valid': True,
        'issues': [],
        'warnings': [],
        'metrics': {}
    }
    
    if not material.get('csv_available', False):
        results['skipped'] = True
        results['reason'] = 'No CSV data available'
        return results
    
    csv_file = materials_dir / material['data_file']
    
    if not csv_file.exists():
        results['valid'] = False
        results['issues'].append(f"ERROR: CSV file not found: {csv_file}")
        return results
    
    try:
        wavelength_nm, n, k = load_material_csv(csv_file)
    except Exception as e:
        results['valid'] = False
        results['issues'].append(f"ERROR: Failed to load CSV: {e}")
        return results
    
    # Basic metrics
    results['metrics']['wavelength_range_nm'] = [float(wavelength_nm.min()), float(wavelength_nm.max())]
    results['metrics']['n_range'] = [float(n.min()), float(n.max())]
    results['metrics']['k_range'] = [float(k.min()), float(k.max())]
    results['metrics']['num_points'] = len(wavelength_nm)
    
    all_issues = []
    
    # Check physical bounds
    material_type = material.get('material_type', 'unknown')
    ok, issues = check_physical_bounds(wavelength_nm, n, k, material_type)
    all_issues.extend(issues)
    if not ok:
        results['valid'] = False
    
    # Check continuity
    ok, issues = check_continuity(wavelength_nm, n, k)
    all_issues.extend(issues)
    
    # Check KK consistency
    ok, issues = check_kramers_kronig_consistency(wavelength_nm, n, k)
    all_issues.extend(issues)
    
    # Check fit accuracy
    fit_params = material.get('drude_lorentz_fit', {})
    ok, issues, fit_metrics = check_fit_accuracy(wavelength_nm, n, k, fit_params)
    all_issues.extend(issues)
    results['metrics']['fit'] = fit_metrics
    
    # Separate errors from warnings
    results['issues'] = [i for i in all_issues if 'ERROR' in i]
    results['warnings'] = [i for i in all_issues if 'WARNING' in i or 'INFO' in i]
    
    if verbose:
        status = '✓' if results['valid'] and len(results['warnings']) == 0 else \
                 '⚠' if results['valid'] else '✗'
        print(f"  {status} {mat_id}")
        for issue in results['issues']:
            print(f"      {issue}")
        for warning in results['warnings'][:3]:
            print(f"      {warning}")
        if len(results['warnings']) > 3:
            print(f"      ... and {len(results['warnings']) - 3} more warnings")
    
    return results


def validate_all_materials(
    materials_dir: Path,
    specific_material: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Validate all materials in the database.
    
    Returns comprehensive validation report.
    """
    with open(materials_dir / 'index.json') as f:
        index = json.load(f)
    
    if verbose:
        print(f"\nValidating materials database v{index.get('version', 'unknown')}")
        print("=" * 60)
    
    results = {
        'version': index.get('version', 'unknown'),
        'materials': {}
    }
    
    for material in index['materials']:
        mat_id = material['material_id']
        
        if specific_material and mat_id != specific_material:
            continue
        
        result = validate_material(material, materials_dir, verbose)
        results['materials'][mat_id] = result
    
    # Summary statistics
    all_results = list(results['materials'].values())
    results['summary'] = {
        'total': len(all_results),
        'valid': sum(1 for r in all_results if r.get('valid', False)),
        'with_warnings': sum(1 for r in all_results if len(r.get('warnings', [])) > 0),
        'skipped': sum(1 for r in all_results if r.get('skipped', False)),
        'failed': sum(1 for r in all_results if not r.get('valid', True) and not r.get('skipped', False))
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"  Total materials:  {results['summary']['total']}")
        print(f"  Valid:            {results['summary']['valid']}")
        print(f"  With warnings:    {results['summary']['with_warnings']}")
        print(f"  Skipped (no CSV): {results['summary']['skipped']}")
        print(f"  Failed:           {results['summary']['failed']}")
    
    return results


def generate_comparison_plots(
    materials_dir: Path,
    output_dir: Optional[Path] = None,
    specific_material: Optional[str] = None
):
    """
    Generate comparison plots between tabulated data and fits.
    
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib required for plotting. Install with: pip install matplotlib")
        return
    
    if output_dir is None:
        output_dir = materials_dir / 'validation_plots'
    output_dir.mkdir(exist_ok=True)
    
    with open(materials_dir / 'index.json') as f:
        index = json.load(f)
    
    for material in index['materials']:
        mat_id = material['material_id']
        
        if specific_material and mat_id != specific_material:
            continue
        
        if not material.get('csv_available', False):
            continue
        
        csv_file = materials_dir / material['data_file']
        if not csv_file.exists():
            continue
        
        try:
            wavelength_nm, n, k = load_material_csv(csv_file)
        except Exception:
            continue
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{material.get('name', mat_id)}", fontsize=14, fontweight='bold')
        
        # Plot n, k vs wavelength
        ax1 = axes[0, 0]
        ax1.plot(wavelength_nm, n, 'b-', label='n (tabulated)', linewidth=1.5)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('n', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(wavelength_nm, k, 'r-', label='k (tabulated)', linewidth=1.5)
        ax1_twin.set_ylabel('k', color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        ax1.set_title('Refractive Index')
        ax1.grid(True, alpha=0.3)
        
        # Plot permittivity
        eps1, eps2 = compute_epsilon_from_nk(n, k)
        ax2 = axes[0, 1]
        ax2.plot(wavelength_nm, eps1, 'b-', label='ε₁', linewidth=1.5)
        ax2.plot(wavelength_nm, eps2, 'r-', label='ε₂', linewidth=1.5)
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Permittivity')
        ax2.legend()
        ax2.set_title('Complex Permittivity')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Plot fit comparison if available
        fit_params = material.get('drude_lorentz_fit', {})
        if fit_params and (fit_params.get('drude_terms') or fit_params.get('lorentz_terms')):
            energy_eV = wavelength_to_energy_eV(wavelength_nm)
            eps1_fit, eps2_fit = compute_drude_lorentz_epsilon(
                energy_eV,
                fit_params.get('eps_inf', 1.0),
                fit_params.get('drude_terms', []),
                fit_params.get('lorentz_terms', [])
            )
            
            ax3 = axes[1, 0]
            ax3.plot(wavelength_nm, eps1, 'b-', label='ε₁ (data)', linewidth=1.5)
            ax3.plot(wavelength_nm, eps1_fit, 'b--', label='ε₁ (fit)', linewidth=1.5)
            ax3.set_xlabel('Wavelength (nm)')
            ax3.set_ylabel('ε₁')
            ax3.legend()
            ax3.set_title('ε₁: Data vs Fit')
            ax3.grid(True, alpha=0.3)
            
            ax4 = axes[1, 1]
            ax4.plot(wavelength_nm, eps2, 'r-', label='ε₂ (data)', linewidth=1.5)
            ax4.plot(wavelength_nm, eps2_fit, 'r--', label='ε₂ (fit)', linewidth=1.5)
            ax4.set_xlabel('Wavelength (nm)')
            ax4.set_ylabel('ε₂')
            ax4.legend()
            ax4.set_title('ε₂: Data vs Fit')
            ax4.grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Drude-Lorentz fit available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('ε₁: Data vs Fit')
            axes[1, 1].text(0.5, 0.5, 'No Drude-Lorentz fit available',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('ε₂: Data vs Fit')
        
        plt.tight_layout()
        
        output_file = output_dir / f'{mat_id}_validation.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Validate optical materials database')
    parser.add_argument('material', nargs='?', help='Specific material ID to validate')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
    parser.add_argument('--report', action='store_true', help='Save JSON report')
    parser.add_argument('--dir', default='materials', help='Materials directory path')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    materials_dir = Path(args.dir)
    if not materials_dir.exists():
        # Try relative to script location
        materials_dir = Path(__file__).parent
    
    if not (materials_dir / 'index.json').exists():
        print(f"ERROR: index.json not found in {materials_dir}")
        return 1
    
    # Run validation
    results = validate_all_materials(
        materials_dir,
        specific_material=args.material,
        verbose=not args.quiet
    )
    
    # Save report if requested
    if args.report:
        report_file = materials_dir / 'validation_report.json'
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nReport saved to: {report_file}")
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating comparison plots...")
        generate_comparison_plots(materials_dir, specific_material=args.material)
    
    # Return exit code
    return 0 if results['summary']['failed'] == 0 else 1


if __name__ == '__main__':
    exit(main())

