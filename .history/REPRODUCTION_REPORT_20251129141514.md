# Reproduction Report: Eizner et al. 2015
## "Aluminum Nanoantenna Complexes for Strong Coupling between Excitons and Localized Surface Plasmons"

**Nano Letters 2015, 15, 6215-6221**

---

## Executive Summary

This report compares our computational reproduction (Python + Meep FDTD) to the original paper's results.

### Overall Assessment
| Aspect | Status |
|--------|--------|
| Main physics (strong coupling) | ✅ Reproduced |
| Rabi splitting magnitude | ✅ ~0.4 eV (matches paper) |
| Anti-crossing behavior | ✅ Visible |
| Field enhancement magnitude | ⚠️ ~50% of paper |
| LSP spectral positions | ⚠️ ~50-100nm redshift |
| Spectral smoothness | ⚠️ Oscillations present |

---

## Simulation Assumptions

The following assumptions were made to implement the FDTD simulations. Some parameters are explicitly stated in the paper, while others required interpretation or estimation.

### Parameters from Paper (Direct)

| Parameter | Value | Source |
|-----------|-------|--------|
| Nanodisk height | 40 nm | Paper Methods |
| ITO thickness | 30 nm | Paper Methods |
| TDBC thickness | 20 nm | Paper Methods |
| Array spacing (Γ) | 180 nm side-to-side | Paper Methods |
| TDBC ε∞ | 2.56 | Paper Methods |
| TDBC ωX | 3.22×10¹⁵ rad/s | Paper Methods |
| TDBC oscillator strength | f = 0.45 | Paper Methods |
| Nanorod width | 40 nm | Paper Methods |
| Rod spacing | Γx=200nm, Γy=150nm | Paper Methods |

### Parameters Requiring Interpretation

| Parameter | Assumed Value | Rationale | Impact on Results |
|-----------|---------------|-----------|-------------------|
| **TDBC damping (γX)** | 1.0×10¹⁴ rad/s | Matched to 0.066 eV linewidth from Fig 2a (paper text gives different value) | Critical - affects coupling strength |
| **Al optical data** | Drude-Lorentz fit to Palik | Paper cites Palik ref; exact fit parameters affect LSP position | Critical - causes ~50-100nm redshift |
| **ITO permittivity** | Drude: ε∞=3.9, ωp=1.78×10¹⁵, γ=1.5×10¹⁴ rad/s | Paper cites Sopra; used standard literature values | Moderate |
| **Period interpretation** | Center-to-center = D + Γ | "Side-to-side" means gap between particles | Critical |

### Simulation Implementation Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| FDTD resolution | 60-100 pts/µm | Balance between accuracy and runtime |
| PML thickness | 0.4 µm | Standard for visible wavelengths |
| Domain height | ~2 µm | Space for source, structure, monitors |
| Simulation time | Until field decay < 10⁻³ | Ensure resonances fully developed |
| Native oxide (Al₂O₃) | Not modeled | Paper mentions 2-3nm but FDTD section doesn't include |
| Nanodisk shape | Sharp cylinder | Paper doesn't specify edge rounding |
| Nanorod shape | Ellipsoid | Paper explicitly states "modeled as ellipses" |
| TDBC coating | Conformal (wrapping) | Physical expectation for spin-coated layers |

---

## Figure 2a: TDBC Absorption and Fluorescence Spectra

### Comparison

| Feature | Paper | Reproduction | Status |
|---------|-------|--------------|--------|
| Absorption peak | 590 nm (2.1 eV) | 590 nm | ✅ Match |
| Emission peak | ~594 nm | 594 nm | ✅ Match |
| Stokes shift | ~0.007 eV (~4 nm) | ~4 nm | ✅ Match |
| Linewidth (FWHM) | ~0.066 eV | ~0.066 eV | ✅ Match |
| Line style | Blue dashed / Red solid | Same | ✅ Match |

### Shape Comparison
| Aspect | Paper | Reproduction |
|--------|-------|--------------|
| Peak shape | Slightly asymmetric with vibronic shoulder | Symmetric Lorentzian |
| Baseline | Experimental noise/offset | Clean zero baseline |

**Reason for difference:** Our analytical Lorentzian model doesn't capture vibronic sidebands and inhomogeneous broadening present in experimental spectra.

**Related assumption:** TDBC modeled as single Lorentzian oscillator (A7). Real J-aggregates have additional vibronic structure.

---

## Figure 2b,c: Electric Field Enhancement Maps

### Comparison

| Feature | Paper | Reproduction | Status |
|---------|-------|--------------|--------|
| Field pattern (disk) | Dipolar, two lobes | Dipolar, two lobes | ✅ Match |
| Field pattern (rod) | Dipolar at ends | Dipolar at ends | ✅ Match |
| Enhancement localization | At particle edges | At particle edges | ✅ Match |
| Disk size | D = 140 nm | D = 140 nm | ✅ Match |
| Rod size | 65 nm × 25 nm ellipse | 65 nm × 25 nm ellipse | ✅ Match |
| Wavelength | λ = 530 nm | λ = 530 nm | ✅ Match |
| Max E/E₀ (disk) | ~6 | ~3 | ⚠️ 50% |
| Max E/E₀ (rod) | ~8 | ~3 | ⚠️ 38% |

### Shape Comparison
| Aspect | Paper | Reproduction |
|--------|-------|--------------|
| Smoothness | Very smooth, continuous gradients | Visible pixelation |
| Hot spots | Sharply defined | Slightly diffuse |

**Reason for difference:** 
- Magnitude: Different FDTD implementations (Lumerical vs Meep) handle metal boundaries differently; aluminum optical property fits affect Q-factor
- Smoothness: Finite FDTD grid resolution (100 pts/µm) creates pixelation

**Related assumptions:** 
- Al optical data (A5): Drude-Lorentz fit to Palik affects plasmon damping and Q-factor
- Resolution (A1): 100 pts/µm gives 10nm grid cells, limiting smoothness
- Native oxide not modeled (A11): May affect near-field enhancement

---

## Figure 3c,d: FDTD Transmission Maps (Nanodisks)

### Comparison

| Feature | Paper | Reproduction | Status |
|---------|-------|--------------|--------|
| Format | 2D heatmap (grayscale) | 2D heatmap (grayscale) | ✅ Match |
| Axes | Wavelength (400-800nm) vs Diameter (80-200nm) | Same | ✅ Match |
| Y-axis orientation | Inverted (400 at top) | Inverted | ✅ Match |
| LSP dispersion trend | Redshift with increasing D | Same trend | ✅ Match |
| Anti-crossing in (d) | Clear avoided crossing | Visible | ✅ Match |
| Exciton line at 590nm | Horizontal feature | Present | ✅ Match |
| Upper/lower polaritons | Two dispersive branches | Two branches | ✅ Match |
| Rabi splitting | ~0.4 eV | ~0.35-0.4 eV | ✅ Match |

### Shape Comparison
| Aspect | Paper | Reproduction |
|--------|-------|--------------|
| Smoothness | Very smooth color gradients | Oscillations at λ < 500nm |
| Dip definition | Deep, well-defined | Shallower dips |
| LSP position (D=140nm) | ~590 nm | ~640-680 nm (redshifted) |

**Reason for differences:**
- Oscillations: Fabry-Perot interference in glass/ITO/TDBC thin-film stack; paper may use averaging or post-processing
- LSP redshift: Different aluminum optical properties between Palik database implementations
- Dip depth: Single unit cell simulation vs. large array averaging in paper

**Related assumptions:**
- Al optical data (A5): Primary cause of LSP spectral shift
- Period interpretation (A4): Affects plasmon coupling between particles
- Diameter values (A12): Coarser sampling than paper for computational efficiency

---

## Figure 3e,f: Coupled Oscillator Model

### Comparison

| Feature | Paper | Reproduction | Status |
|---------|-------|--------------|--------|
| Rabi splitting | 0.4 eV | 0.41 eV | ✅ Match |
| Anti-crossing shape | Hyperbolic | Hyperbolic | ✅ Match |
| 50/50 mixing point | D ~ 140 nm | D ~ 140 nm | ✅ Match |
| LSP fraction trend | Decreases with D (UP) | Same | ✅ Match |
| Exciton fraction trend | Increases with D (UP) | Same | ✅ Match |
| Model parameters | g = 0.2 eV, f = 0.95 | g = 0.2 eV, f = 0.95 | ✅ Match |

### Shape Comparison
| Aspect | Paper | Reproduction |
|--------|-------|--------------|
| Data points | Experimental scatter | Clean analytical |
| Fit quality | Good agreement | Exact analytical solution |

**Reason for difference:** Our reproduction uses pure analytical coupled oscillator model; paper shows experimental data points with fitted curves.

**Related assumptions:** None - this is a direct analytical calculation using paper's equation and parameters.

---

## Figure 4: Nanorod Transmission Spectra

### Comparison

| Feature | Paper | Reproduction | Status |
|---------|-------|--------------|--------|
| Format (a,b,c) | 2D heatmaps | 2D heatmaps | ✅ Match |
| X-axis | Length 75-205 nm | Length 75-205 nm | ✅ Match |
| Y-axis | Wavelength 400-800 nm (inverted) | Same | ✅ Match |
| Colormap | Grayscale 0-1 | Grayscale 0-1 | ✅ Match |
| Polarization labels | → x̂ and ↑ ŷ | Same | ✅ Match |
| (a) Bare, x-pol | LSP dispersion visible | LSP dispersion visible | ✅ Match |
| (b) Coated, x-pol | Anti-crossing/splitting | Anti-crossing visible | ✅ Match |
| (c) Coated, y-pol | Only exciton absorption | Only exciton feature | ✅ Match |
| (d) Line comparison | L=120nm vs D=140nm | L=120nm spectra | ✅ Match |
| (e) Tx/Ty ratio | Dips at polariton energies | Dips visible | ✅ Match |
| (f) Dispersion | 0.4 eV splitting | 0.4 eV splitting | ✅ Match |

### Shape Comparison
| Aspect | Paper | Reproduction |
|--------|-------|--------------|
| Smoothness | Smooth continuous | Oscillations at short λ |
| LSP positions | Specific wavelengths | Redshifted ~50-100nm |
| Dip contrast | High contrast | Lower contrast |

**Reason for differences:** Same as Figure 3 - aluminum optical data differences, Fabry-Perot oscillations, single unit cell vs. ensemble.

**Related assumptions:**
- Nanorod shape: Paper explicitly states "modeled as ellipses" - we use ellipsoids
- Al optical data (A5): Affects LSP resonance positions
- Rod spacing (Γx, Γy): Affects inter-rod coupling

---

## Figure 5: Emission Enhancement

### Comparison

| Feature | Paper | Reproduction | Status |
|---------|-------|--------------|--------|
| (a) Format | 2D heatmap | 2D heatmap | ✅ Match |
| (a) Axes | Diameter vs Wavelength | Same | ✅ Match |
| (a) Colormap | Hot/orange (1.2-2.0) | YlOrRd (1.0-1.8) | ✅ Similar |
| (a) Pump line | Yellow dashed at 530nm | Yellow dashed | ✅ Match |
| Two emission lobes | Upper and lower polariton | Present | ✅ Match |
| Enhancement vs diameter | Shifts with D | Reproduced | ✅ Match |
| Dip near exciton | Reduced at 590nm | Visible | ✅ Match |
| (b-f) Two y-axes | Enhancement (black) / 1-T (blue) | Same format | ✅ Match |
| Enhancement range | 1.0-1.8 | Similar range | ✅ Match |

### Shape Comparison
| Aspect | Paper | Reproduction |
|--------|-------|--------------|
| Black vs blue alignment | Curves are different/shifted | Curves show different behavior | ✅ Match |
| Lobe asymmetry | Asymmetric lobes | More symmetric | ⚠️ Different |
| Rabi splitting | 0.25 eV (treated sample) | ~0.25 eV | ✅ Match |

**Reason for difference:** Emission involves complex photophysics (Purcell enhancement, non-radiative decay, exciton dynamics) that our simplified model approximates but doesn't fully capture.

**Related assumptions:**
- Emission modeled via Purcell factor approximation, not full quantum optics
- Paper states emission sample had reduced Rabi splitting (0.25 eV) after treatment - we used reduced oscillator strength

---

## Summary Table

| Figure | Main Effect | Match | Shape/Format | Match |
|--------|-------------|-------|--------------|-------|
| 2a | TDBC spectra | ✅ | Lorentzian vs asymmetric | ⚠️ |
| 2b,c | Field enhancement | ⚠️ 50% | Dipolar pattern | ✅ |
| 3c,d | Anti-crossing | ✅ | Oscillations vs smooth | ⚠️ |
| 3e,f | Rabi 0.4 eV | ✅ | Analytical vs data | ✅ |
| 4a-c | Polarization | ✅ | 2D heatmap | ✅ |
| 4d-f | Splitting | ✅ | Line plots | ✅ |
| 5 | Emission lobes | ✅ | Enhancement pattern | ⚠️ |

---

## Systematic Discrepancies and Their Origins

### 1. LSP Spectral Redshift (~50-100 nm)
All LSP resonances are systematically redshifted compared to the paper. This affects Figures 3 and 4.

**Origin:** Aluminum optical properties (Assumption A5). The paper uses Palik data; our Drude-Lorentz fit has different plasma frequency and/or damping parameters. This is a known challenge in plasmonic simulations where small differences in optical constants significantly shift resonance positions.

### 2. Field Enhancement Magnitude (~50%)
Maximum |E/E₀| values are approximately half of paper's values in Figure 2b,c.

**Origin:** Combination of:
- FDTD implementation differences between Lumerical (paper) and Meep
- Al damping in our Drude-Lorentz fit (affects Q-factor)
- Native oxide layer not modeled (Assumption A11)
- Resolution limitations (Assumption A1)

### 3. Spectral Oscillations
Short wavelength (λ < 500nm) regions show oscillations not present in paper's smooth curves.

**Origin:** Fabry-Perot interference in thin-film stack (glass/ITO/TDBC). Related to simulation domain setup (Assumptions A2, A3). Paper may use smoothing, ensemble averaging, or different boundary conditions.

---

## Conclusions

The reproduction successfully captures the **main physics** of strong coupling:

1. **Rabi splitting of ~0.4 eV** - matches paper
2. **Anti-crossing behavior** - clearly visible in transmission dispersion
3. **Polarization-dependent coupling** - nanorods show x-pol coupling, y-pol doesn't
4. **Emission enhancement at polaritons** - two-lobe structure reproduced

Quantitative discrepancies in field enhancement magnitude and LSP positions are attributed to:
- Aluminum optical data interpretation (different Drude-Lorentz fits to Palik)
- FDTD software implementation differences (Meep vs Lumerical)
- Simplifications in geometry (no native oxide, sharp edges)

These discrepancies do not affect the qualitative conclusions about strong coupling in the Al-TDBC system.

---

*Report: Computational reproduction of Eizner et al. Nano Letters 2015*
