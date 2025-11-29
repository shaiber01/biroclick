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

### Parameters Requiring Interpretation

| Parameter | Assumed Value | Rationale | Impact |
|-----------|---------------|-----------|--------|
| TDBC damping (γX) | 1.0×10¹⁴ rad/s | Matched to 0.066 eV linewidth from Fig 2a | Critical |
| Al optical data | Drude-Lorentz fit to Palik | Paper cites Palik; exact fit affects LSP position | Critical |
| ITO permittivity | Drude model from literature | Paper cites Sopra database | Moderate |
| Period interpretation | Center-to-center = D + Γ | "Side-to-side" = gap between particles | Critical |

### Simulation Implementation

| Parameter | Value |
|-----------|-------|
| FDTD resolution | 60-100 pts/µm |
| Native oxide | Not modeled |
| Nanodisk shape | Sharp cylinder |
| Nanorod shape | Ellipsoid (as paper states) |
| TDBC coating | Conformal (wrapping) |

---

## Figure 2a: TDBC Absorption and Fluorescence Spectra

<table>
<tr>
<th>Original Paper</th>
<th>Reproduction</th>
</tr>
<tr>
<td><img src="https://web-api.textin.com/ocr_image/external/da2a81b4fa93d420.jpg" width="350"/></td>
<td><img src="fig2a_exact.png" width="350"/></td>
</tr>
</table>

### Comparison

| Feature | Paper | Reproduction | Status |
|---------|-------|--------------|--------|
| Absorption peak | 590 nm (2.1 eV) | 590 nm | ✅ Match |
| Emission peak | ~594 nm | 594 nm | ✅ Match |
| Stokes shift | ~0.007 eV | ~4 nm | ✅ Match |
| Linewidth (FWHM) | ~0.066 eV | ~0.066 eV | ✅ Match |

### Shape Comparison
| Aspect | Paper | Reproduction |
|--------|-------|--------------|
| Peak shape | Asymmetric with vibronic shoulder | Symmetric Lorentzian |
| Baseline | Experimental noise | Clean zero baseline |

**Reason for difference:** Analytical Lorentzian model doesn't capture vibronic sidebands present in experimental spectra.

---

## Figure 2b,c: Electric Field Enhancement Maps

<table>
<tr>
<th>Original Paper (2b - Disk)</th>
<th>Original Paper (2c - Rod)</th>
</tr>
<tr>
<td><img src="https://web-api.textin.com/ocr_image/external/3661ce7214790f30.jpg" width="300"/></td>
<td><img src="https://web-api.textin.com/ocr_image/external/168ffd6a9b031803.jpg" width="300"/></td>
</tr>
<tr>
<th colspan="2">Reproduction (2b - Disk, 2c - Rod)</th>
</tr>
<tr>
<td colspan="2"><img src="fig2bc_corrected.png" width="700"/></td>
</tr>
</table>

### Comparison

| Feature | Paper | Reproduction | Status |
|---------|-------|--------------|--------|
| Field pattern (disk) | Dipolar, two lobes | Dipolar, two lobes | ✅ Match |
| Field pattern (rod) | Dipolar at ends | Dipolar at ends | ✅ Match |
| Disk size | D = 140 nm | D = 140 nm | ✅ Match |
| Rod size | 65×25 nm ellipse | 65×25 nm ellipse | ✅ Match |
| Max E/E₀ (disk) | ~6 | ~3 | ⚠️ 50% |
| Max E/E₀ (rod) | ~8 | ~3 | ⚠️ 38% |

### Shape Comparison
| Aspect | Paper | Reproduction |
|--------|-------|--------------|
| Smoothness | Very smooth gradients | Visible pixelation |
| Hot spots | Sharply defined | Slightly diffuse |

**Reason for difference:** FDTD implementation differences (Lumerical vs Meep); aluminum optical property fits affect Q-factor; finite grid resolution.

---

## Figure 3c,d: FDTD Transmission Maps (Nanodisks)

<table>
<tr>
<th>Original Paper (3c - Bare)</th>
<th>Original Paper (3d - Coated)</th>
</tr>
<tr>
<td><img src="https://web-api.textin.com/ocr_image/external/cf174ff4fac4b75f.jpg" width="300"/></td>
<td><img src="https://web-api.textin.com/ocr_image/external/1078e7bb3bd6551b.jpg" width="300"/></td>
</tr>
<tr>
<th colspan="2">Reproduction (3c - Bare, 3d - Coated)</th>
</tr>
<tr>
<td colspan="2"><img src="fig3cd_fast.png" width="700"/></td>
</tr>
</table>

### Comparison

| Feature | Paper | Reproduction | Status |
|---------|-------|--------------|--------|
| LSP dispersion trend | Redshift with increasing D | Same trend | ✅ Match |
| Anti-crossing in (d) | Clear avoided crossing | Visible | ✅ Match |
| Exciton line at 590nm | Horizontal feature | Present | ✅ Match |
| Rabi splitting | ~0.4 eV | ~0.35-0.4 eV | ✅ Match |

### Shape Comparison
| Aspect | Paper | Reproduction |
|--------|-------|--------------|
| Smoothness | Very smooth | Oscillations at λ < 500nm |
| LSP position (D=140nm) | ~590 nm | ~640-680 nm (redshifted) |
| Dip definition | Deep, well-defined | Shallower dips |

**Reason for differences:** Fabry-Perot interference causes oscillations; different Al optical data causes redshift; single unit cell vs. ensemble averaging.

---

## Figure 3e,f: Coupled Oscillator Model

<table>
<tr>
<th>Original Paper (3e)</th>
<th>Original Paper (3f - upper)</th>
<th>Original Paper (3f - lower)</th>
</tr>
<tr>
<td><img src="https://web-api.textin.com/ocr_image/external/6e70e73edacf3324.jpg" width="220"/></td>
<td><img src="https://web-api.textin.com/ocr_image/external/8671242f00fdef43.jpg" width="220"/></td>
<td><img src="https://web-api.textin.com/ocr_image/external/7d2b3e43bf3756c0.jpg" width="220"/></td>
</tr>
<tr>
<th colspan="3">Reproduction (3e, 3f)</th>
</tr>
<tr>
<td colspan="3"><img src="fig3ef_exact.png" width="700"/></td>
</tr>
</table>

### Comparison

| Feature | Paper | Reproduction | Status |
|---------|-------|--------------|--------|
| Rabi splitting | 0.4 eV | 0.41 eV | ✅ Match |
| Anti-crossing shape | Hyperbolic | Hyperbolic | ✅ Match |
| 50/50 mixing point | D ~ 140 nm | D ~ 140 nm | ✅ Match |
| Model parameters | g=0.2eV, f=0.95 | g=0.2eV, f=0.95 | ✅ Match |

### Shape Comparison
| Aspect | Paper | Reproduction |
|--------|-------|--------------|
| Data points | Experimental scatter | Clean analytical |

**Reason for difference:** Pure analytical model vs. experimental data with fitted curves.

---

## Figure 4: Nanorod Transmission Spectra

<table>
<tr>
<th>Original Paper (4a)</th>
<th>Original Paper (4b)</th>
<th>Original Paper (4c)</th>
</tr>
<tr>
<td><img src="https://web-api.textin.com/ocr_image/external/74248f8a246fb35b.jpg" width="220"/></td>
<td><img src="https://web-api.textin.com/ocr_image/external/a3cdd197748a4396.jpg" width="220"/></td>
<td><img src="https://web-api.textin.com/ocr_image/external/6a736599a6693d2d.jpg" width="220"/></td>
</tr>
<tr>
<th>Original Paper (4d)</th>
<th>Original Paper (4e)</th>
<th>Original Paper (4f)</th>
</tr>
<tr>
<td><img src="https://web-api.textin.com/ocr_image/external/a8a9df184a280b48.jpg" width="220"/></td>
<td><img src="https://web-api.textin.com/ocr_image/external/57b5d66080cbe8b3.jpg" width="220"/></td>
<td><img src="https://web-api.textin.com/ocr_image/external/8a646c1307ba4430.jpg" width="220"/></td>
</tr>
<tr>
<th colspan="3">Reproduction (4a-f)</th>
</tr>
<tr>
<td colspan="3"><img src="fig4_reproduction.png" width="900"/></td>
</tr>
</table>

### Comparison

| Feature | Paper | Reproduction | Status |
|---------|-------|--------------|--------|
| Format (a,b,c) | 2D heatmaps | 2D heatmaps | ✅ Match |
| (a) Bare, x-pol | LSP dispersion | LSP dispersion | ✅ Match |
| (b) Coated, x-pol | Anti-crossing | Anti-crossing | ✅ Match |
| (c) Coated, y-pol | Only exciton | Only exciton | ✅ Match |
| (d) Comparison | x vs y polarization | x vs y polarization | ✅ Match |
| (e) Tx/Ty ratio | Polariton dips | Dips visible | ✅ Match |
| (f) Dispersion | 0.4 eV splitting | 0.4 eV splitting | ✅ Match |

### Shape Comparison
| Aspect | Paper | Reproduction |
|--------|-------|--------------|
| Smoothness | Smooth continuous | Oscillations at short λ |
| LSP positions | Specific wavelengths | Redshifted ~50-100nm |

**Reason for differences:** Same as Figure 3 - Al optical data, Fabry-Perot oscillations.

---

## Figure 5: Emission Enhancement

<table>
<tr>
<th>Original Paper (5a)</th>
<th>Original Paper (5b)</th>
<th>Original Paper (5c)</th>
</tr>
<tr>
<td><img src="https://web-api.textin.com/ocr_image/external/980975a27b9043d1.jpg" width="220"/></td>
<td><img src="https://web-api.textin.com/ocr_image/external/21eae1341b2aa7b3.jpg" width="220"/></td>
<td><img src="https://web-api.textin.com/ocr_image/external/5412bb74e2b5125f.jpg" width="220"/></td>
</tr>
<tr>
<th>Original Paper (5d)</th>
<th>Original Paper (5e)</th>
<th>Original Paper (5f)</th>
</tr>
<tr>
<td><img src="https://web-api.textin.com/ocr_image/external/f2382ff214ac8124.jpg" width="220"/></td>
<td><img src="https://web-api.textin.com/ocr_image/external/1dc648930eac3ac1.jpg" width="220"/></td>
<td><img src="https://web-api.textin.com/ocr_image/external/b9bfdb4096a73977.jpg" width="220"/></td>
</tr>
<tr>
<th colspan="3">Reproduction (5a-f)</th>
</tr>
<tr>
<td colspan="3"><img src="fig5_proper.png" width="900"/></td>
</tr>
</table>

### Comparison

| Feature | Paper | Reproduction | Status |
|---------|-------|--------------|--------|
| (a) 2D heatmap format | Yes | Yes | ✅ Match |
| Two emission lobes | Upper and lower polariton | Present | ✅ Match |
| Enhancement vs diameter | Shifts with D | Reproduced | ✅ Match |
| Dip near exciton | Reduced at 590nm | Visible | ✅ Match |
| (b-f) Two y-axes | Enhancement / 1-T | Same format | ✅ Match |
| Black vs blue lines | Different curves | Different curves | ✅ Match |
| Rabi splitting | 0.25 eV | ~0.25 eV | ✅ Match |

### Shape Comparison
| Aspect | Paper | Reproduction |
|--------|-------|--------------|
| Lobe asymmetry | Asymmetric lobes | More symmetric |
| Color range | 1.2-2.0 | 1.0-1.8 |

**Reason for difference:** Emission involves complex photophysics (Purcell enhancement, non-radiative decay) that our simplified model approximates.

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

## Systematic Discrepancies

### 1. LSP Spectral Redshift (~50-100 nm)
All LSP resonances are redshifted compared to the paper.

**Origin:** Aluminum optical properties - different Drude-Lorentz fit parameters.

### 2. Field Enhancement Magnitude (~50%)
Maximum |E/E₀| is approximately half of paper's values.

**Origin:** FDTD implementation differences (Lumerical vs Meep); Al damping parameters.

### 3. Spectral Oscillations
Short wavelength regions show oscillations not in paper.

**Origin:** Fabry-Perot interference in thin-film stack; paper may use smoothing.

---

## Conclusions

The reproduction successfully captures the **main physics** of strong coupling:

1. **Rabi splitting of ~0.4 eV** - matches paper
2. **Anti-crossing behavior** - clearly visible
3. **Polarization-dependent coupling** - nanorods show x-pol coupling only
4. **Emission enhancement at polaritons** - two-lobe structure reproduced

Quantitative discrepancies don't affect the qualitative conclusions about strong coupling in the Al-TDBC system.

---

*Report: Computational reproduction of Eizner et al. Nano Letters 2015*
