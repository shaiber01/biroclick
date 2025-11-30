# Reproduction Report Template

This template defines the structure for `REPRODUCTION_REPORT_<paper_id>.md` files.

---

## Template Structure

```markdown
# Reproduction Report: [Authors] [Year]
## "[Paper Title]"

**[Journal] [Year], [Volume], [Pages]**

---

## Executive Summary

This report compares our computational reproduction (Python + Meep FDTD) to the original paper's results.

### Overall Assessment
| Aspect | Status |
|--------|--------|
| [Main physics phenomenon] | ✅/⚠️/❌ [Status] |
| [Key quantitative result 1] | ✅/⚠️/❌ [Status] |
| [Key quantitative result 2] | ✅/⚠️/❌ [Status] |
| [Secondary effect 1] | ✅/⚠️/❌ [Status] |
| [Secondary effect 2] | ✅/⚠️/❌ [Status] |

---

## Simulation Assumptions

### Parameters from Paper (Direct)

| Parameter | Value | Source |
|-----------|-------|--------|
| [Param 1] | [Value + units] | Paper [Section] |
| [Param 2] | [Value + units] | Paper [Section] |
| ... | ... | ... |

### Parameters Requiring Interpretation

| Parameter | Assumed Value | Rationale | Impact |
|-----------|---------------|-----------|--------|
| [Param 1] | [Value] | [Why this choice] | Critical/Moderate/Minor |
| [Param 2] | [Value] | [Why this choice] | Critical/Moderate/Minor |
| ... | ... | ... | ... |

### Simulation Implementation

| Parameter | Value |
|-----------|-------|
| FDTD resolution | [X] pts/µm |
| [Implementation choice 1] | [Value/description] |
| [Implementation choice 2] | [Value/description] |
| ... | ... |

---

## Figure [X]: [Figure Title]

<table>
<tr>
<th>Original Paper</th>
<th>Reproduction</th>
</tr>
<tr>
<td><img src="[paper_figure_path]" width="350"/></td>
<td><img src="[reproduction_figure_path]" width="350"/></td>
</tr>
</table>

### Comparison

| Feature | Paper | Reproduction | Status |
|---------|-------|--------------|--------|
| [Feature 1] | [Paper value] | [Our value] | ✅/⚠️/❌ |
| [Feature 2] | [Paper value] | [Our value] | ✅/⚠️/❌ |
| ... | ... | ... | ... |

### Shape Comparison

| Aspect | Paper | Reproduction |
|--------|-------|--------------|
| [Aspect 1] | [Paper description] | [Our description] |
| [Aspect 2] | [Paper description] | [Our description] |
| ... | ... | ... |

**Reason for difference:** [Explanation of any discrepancies]

---

[REPEAT FOR EACH FIGURE]

---

## Summary Table

| Figure | Main Effect | Match | Shape/Format | Match |
|--------|-------------|-------|--------------|-------|
| [Fig X] | [Effect] | ✅/⚠️/❌ | [Description] | ✅/⚠️/❌ |
| [Fig Y] | [Effect] | ✅/⚠️/❌ | [Description] | ✅/⚠️/❌ |
| ... | ... | ... | ... | ... |

---

## Systematic Discrepancies

### 1. [Discrepancy Name]
[Description of the systematic discrepancy]

**Origin:** [Technical explanation of why this occurs]

### 2. [Discrepancy Name]
[Description]

**Origin:** [Explanation]

---

## Conclusions

The reproduction successfully captures the **main physics** of [phenomenon]:

1. **[Key finding 1]** - [comparison to paper]
2. **[Key finding 2]** - [comparison to paper]
3. **[Key finding 3]** - [comparison to paper]
4. **[Key finding 4]** - [comparison to paper]

[Final statement about whether quantitative discrepancies affect qualitative conclusions]

---

*Report: Computational reproduction of [Authors] [Journal] [Year]*
```

---

## Status Icons

Use these consistently:
- ✅ = Reproduced / Match
- ⚠️ = Partial / ~50% / Minor discrepancy
- ❌ = Not reproduced / Mismatch / Failed

## Figure Comparison Guidelines

### Side-by-Side Layout

For single figures:
```html
<table>
<tr>
<th>Original Paper</th>
<th>Reproduction</th>
</tr>
<tr>
<td><img src="paper_fig.png" width="350"/></td>
<td><img src="repro_fig.png" width="350"/></td>
</tr>
</table>
```

For multi-panel figures (e.g., 2b and 2c together):
```html
<table>
<tr>
<th>Original Paper (2b - Disk)</th>
<th>Original Paper (2c - Rod)</th>
</tr>
<tr>
<td><img src="paper_2b.png" width="300"/></td>
<td><img src="paper_2c.png" width="300"/></td>
</tr>
<tr>
<th colspan="2">Reproduction (2b - Disk, 2c - Rod)</th>
</tr>
<tr>
<td colspan="2"><img src="repro_2bc.png" width="700"/></td>
</tr>
</table>
```

For 6-panel figures:
```html
<table>
<tr>
<th>Original Paper (4a)</th>
<th>Original Paper (4b)</th>
<th>Original Paper (4c)</th>
</tr>
<tr>
<td><img src="paper_4a.png" width="220"/></td>
<td><img src="paper_4b.png" width="220"/></td>
<td><img src="paper_4c.png" width="220"/></td>
</tr>
<tr>
<th>Original Paper (4d)</th>
<th>Original Paper (4e)</th>
<th>Original Paper (4f)</th>
</tr>
<tr>
<td><img src="paper_4d.png" width="220"/></td>
<td><img src="paper_4e.png" width="220"/></td>
<td><img src="paper_4f.png" width="220"/></td>
</tr>
<tr>
<th colspan="3">Reproduction (4a-f)</th>
</tr>
<tr>
<td colspan="3"><img src="repro_fig4.png" width="900"/></td>
</tr>
</table>
```

## Comparison Table Standards

### Feature Comparison

Always include:
1. **Key quantitative values** (peaks, magnitudes, splitting)
2. **Qualitative patterns** (trends, shapes, features)
3. **Dimensions/sizes** if geometry is shown

Example:
| Feature | Paper | Reproduction | Status |
|---------|-------|--------------|--------|
| Absorption peak | 590 nm (2.1 eV) | 590 nm | ✅ Match |
| Emission peak | ~594 nm | 594 nm | ✅ Match |
| Linewidth (FWHM) | ~0.066 eV | ~0.066 eV | ✅ Match |
| Max E/E₀ (disk) | ~6 | ~3 | ⚠️ 50% |

### Shape Comparison

For qualitative differences that don't fit in the feature table:

| Aspect | Paper | Reproduction |
|--------|-------|--------------|
| Peak shape | Asymmetric with vibronic shoulder | Symmetric Lorentzian |
| Smoothness | Very smooth gradients | Visible pixelation |
| Hot spots | Sharply defined | Slightly diffuse |

## Summary Table Format

Quick reference for all figures:

| Figure | Main Effect | Match | Shape/Format | Match |
|--------|-------------|-------|--------------|-------|
| 2a | TDBC spectra | ✅ | Lorentzian vs asymmetric | ⚠️ |
| 2b,c | Field enhancement | ⚠️ 50% | Dipolar pattern | ✅ |
| 3c,d | Anti-crossing | ✅ | Oscillations vs smooth | ⚠️ |

## Systematic Discrepancies Format

Number and name each systematic issue:

### 1. LSP Spectral Redshift (~50-100 nm)
All LSP resonances are redshifted compared to the paper.

**Origin:** Aluminum optical properties - different Drude-Lorentz fit parameters.

### 2. Field Enhancement Magnitude (~50%)
Maximum |E/E₀| is approximately half of paper's values.

**Origin:** FDTD implementation differences; Al damping parameters.

## Conclusions Format

Use bold for key physics and numbered list:

The reproduction successfully captures the **main physics** of strong coupling:

1. **Rabi splitting of ~0.4 eV** - matches paper
2. **Anti-crossing behavior** - clearly visible
3. **Polarization-dependent coupling** - nanorods show x-pol coupling only

End with statement about whether discrepancies affect conclusions:

> Quantitative discrepancies don't affect the qualitative conclusions about strong coupling in the Al-TDBC system.

