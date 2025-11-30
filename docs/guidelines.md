# Optics Paper Reproduction Guidelines

Lessons learned from reproducing optics/metamaterials papers with FDTD simulations.

## 1. Parameter Extraction

### Cross-Check Everything

Parameters in scientific papers can have errors. Always cross-check:

| Source | Reliability | Notes |
|--------|-------------|-------|
| Figure axes/annotations | Highest | Shows actual data |
| Figure captions | High | Usually reviewed carefully |
| Methods section | Medium | May have typos |
| Main text | Medium | Context-dependent |
| Supplementary | Lower | Less reviewed |

**Rule**: When text and figures disagree by >20%, figures are more reliable.

### Common Parameter Locations

| Parameter Type | Typical Location |
|----------------|------------------|
| Geometry dimensions | Methods, figure captions |
| Material references | Methods, supplementary |
| Wavelength ranges | Figure axes |
| Linewidths | Extract from figure FWHM |
| Refractive indices | Methods or cited reference |

---

## 2. Geometry Interpretation

### Spacing vs Period

This is the most common source of confusion:

```
                Period (center-to-center)
         ◄────────────────────────────────►
         
    ┌────────┐              ┌────────┐
    │        │    Gap       │        │
    │  Disk  │◄──────────►  │  Disk  │
    │        │              │        │
    └────────┘              └────────┘
         
         ◄──────────────────►
              Spacing (usually = gap)
```

**Interpretation Guide**:
- "Spacing" or "separation" → Usually gap (edge-to-edge)
- "Period" or "pitch" → Center-to-center
- Check: Period = Size + Gap

### Shape Ambiguities

| Paper Term | Possible Shapes | How to Resolve |
|------------|-----------------|----------------|
| "Disk" | Cylinder, oblate spheroid | Check SEM, usually cylinder |
| "Rod" | Cylinder, ellipsoid, capsule | Check aspect ratio, SEM |
| "Nanoparticle" | Sphere, hemisphere, rounded cube | Check SEM/TEM |
| "Antenna" | Could be any elongated shape | Check geometry section |

### Layer Conformality

How coatings follow underlying topography:

| Deposition Method | Typical Conformality |
|-------------------|---------------------|
| Spin-coating | Conformal for thin films |
| Evaporation | Line-of-sight, less conformal |
| Sputtering | Semi-conformal |
| ALD | Highly conformal |
| CVD | Conformal |

**Default**: For organic/polymer coatings (like J-aggregates), assume conformal.

---

## 3. Material Models

### Metal Optical Data

Different databases can shift plasmonic resonances by 10-100 nm!

| Database | Coverage | Notes |
|----------|----------|-------|
| Palik | Comprehensive | Most cited, good default |
| Johnson-Christy | Au, Ag | Common for noble metals |
| Rakic (Lorentz-Drude) | Al, Au, Ag, Cu | Analytical model |
| CRC Handbook | Various | Alternative reference |

**Strategy**:
1. Check if paper cites specific reference
2. If not, use Palik as default
3. Document choice as critical assumption
4. If resonance is shifted, try alternative database

### Resonant Materials (J-aggregates, QDs, etc.)

Lorentzian oscillator model:

```
ε(ω) = ε∞ + f·ω₀² / (ω₀² - ω² - iγω)
```

Parameters:
- `ε∞`: Background permittivity (~2.5 for organics)
- `ω₀`: Resonance frequency (from absorption peak)
- `γ`: Damping (≈ FWHM in angular frequency)
- `f`: Oscillator strength (determines coupling strength)

**Critical**: Extract `γ` from paper's absorption spectrum FWHM, don't guess!

### Dielectrics

Usually less critical, but:
- Use dispersive n(λ) for broadband simulations
- Constant n OK for narrow ranges (<100 nm)
- Include native oxides (2-3 nm) if mentioned

---

## 4. Resolution Guidelines

### Minimum Resolution Table

| Simulation Type | Resolution | Reasoning |
|-----------------|------------|-----------|
| Far-field spectra | λ/(10·n_max) | Adequate for spectral features |
| Near-field maps | λ/(20·n_max) | Resolve hot spots |
| Small features | 5+ points across | Geometry accuracy |
| Metal surfaces | 2-5 nm | Skin depth |

Where `n_max` is the highest refractive index in the simulation.

### Convergence Testing

```python
# Run at two resolutions
res1 = baseline_resolution
res2 = 1.5 * baseline_resolution

# Compare key quantities
if abs(peak_wavelength_1 - peak_wavelength_2) / peak_wavelength_1 > 0.05:
    print("Resolution insufficient - increase")
```

**Warning**: If results look WORSE at higher resolution, suspect numerical artifacts.

---

## 5. Quantitative Thresholds

### Discrepancy Classification

| Quantity | Excellent | Acceptable | Investigate |
|----------|-----------|------------|-------------|
| Resonance λ | ±2% | ±5% | >10% |
| Linewidth/Q | ±10% | ±30% | >50% |
| T/R/A | ±5% | ±15% | >30% |
| Field enhancement | ±20% | ±50% | >2× |
| Effective index | ±1% | ±3% | >5% |

### Known Acceptable Discrepancies

These are real physics, not errors:

1. **Fabry-Perot oscillations**: Thin film interference that may be averaged out in experiment
2. **Systematic wavelength shift**: From material data choice (OK if trends correct)
3. **Amplitude differences**: From collection efficiency, normalization

### Failure Indicators

These require investigation:

- Missing features (resonance not appearing)
- Wrong trend (opposite shift direction)
- Order of magnitude differences
- Unphysical results (T > 1, negative absorption)

---

## 6. Validation Hierarchy

### Why Order Matters

```
Material Error ──► Propagates to ALL subsequent stages
                   (shifted resonances, wrong coupling)

Geometry Error ──► Wrong mode structure
                   (missing features, wrong Q-factors)

Numerical Error ─► Usually smaller effect
                   (noisy results, slow convergence)
```

### Mandatory Stages

1. **Material Validation**
   - Plot ε(ω), n(ω), k(ω)
   - Compare to any spectra in paper
   - Validate absorption peaks, linewidths
   
2. **Single Structure**
   - Isolated structure, no arrays
   - Validate resonance position, Q-factor
   - Check mode profile if shown
   
3. **Array/System**
   - Add periodicity, coupling
   - Validate collective effects
   
4. **Parameter Sweeps**
   - Reproduce multi-parameter figures
   - Validate trends (not just points)
   
5. **Complex Physics** (if needed)
   - Only after linear validation passes

---

## 7. Common Pitfalls

### Pitfall 1: Starting Too Complex

**Wrong**: "Let me simulate the full 3D periodic structure with all effects"
**Right**: "Let me validate materials, then single structure, then add complexity"

### Pitfall 2: Ignoring Systematic Shifts

**Wrong**: "Simulation is 50nm off, must be completely wrong"
**Right**: "There's a systematic ~5% shift from Palik Al data, trends are correct"

### Pitfall 3: Chasing Perfect Match

**Wrong**: Spending hours trying to match every pixel
**Right**: Document discrepancies, understand causes, accept "good enough"

### Pitfall 4: Hidden Assumptions

**Wrong**: Silently using literature defaults without documenting
**Right**: Explicitly list every assumption with source and reasoning

### Pitfall 5: 2D vs 3D Mismatch

**Problem**: 2D simulations are faster but may miss 3D effects
**Solution**: Use 2D for validation, note systematic differences, run 3D for final results if needed

---

## 8. Output Best Practices

### Data Files

```
filename: <paper_id>_<stage_id>_<description>.csv

Example: al_nano_stage2_D75_transmission.csv

Contents:
# Paper: aluminum_nanoantenna_2013
# Stage: stage2_bare_disk_sweep
# Description: Transmission spectrum for D=75nm bare disk
# Generated: 2025-11-29T10:15:00Z
wavelength_nm,transmission,reflection,absorption
400,0.82,0.15,0.03
401,0.81,0.16,0.03
...
```

### Plot Requirements

Every plot must have:
1. **Title**: "Stage X – Description – Target: Fig. Y"
2. **Labeled axes** with units
3. **Same axis ranges** as paper figure
4. **Same orientation** (check if wavelength runs high→low)
5. **Legend** if multiple curves

### Comparison Figures

Side-by-side comparison format:
```
┌─────────────────────┬─────────────────────┐
│   Paper Figure      │   Simulation        │
│                     │                     │
│   [Fig. 3a]         │   [Our result]      │
│                     │                     │
└─────────────────────┴─────────────────────┘
Caption: Comparison of Fig. 3a (left) with simulation (right).
Peak positions: Paper 520nm, Simulation 540nm (+3.8%).
```

---

## 9. When to Stop

### Good Stopping Points

✓ Main phenomenon reproduced qualitatively
✓ Quantitative agreement within "acceptable" thresholds
✓ Discrepancies understood and documented
✓ Further improvement needs unavailable information

### Signs of Diminishing Returns

- Last 3 iterations improved <5%
- Stuck on minor features while main claims validated
- Approaching limits of FDTD accuracy
- Changes require unverifiable assumptions

### Documentation Over Perfection

A well-documented partial reproduction is more valuable than:
- An undocumented "perfect" match
- Endless iteration on minor details
- Claims of accuracy without evidence

---

## 10. Report Writing Standards

### Executive Summary Format

Start with overall assessment table:

| Aspect | Status |
|--------|--------|
| Main physics (phenomenon name) | ✅ Reproduced |
| Key quantitative result | ⚠️ ~50% of paper |

### Figure Comparison Format

For EVERY reproduced figure, include:

1. **Side-by-side images** (HTML table format)
2. **Comparison table** (Feature | Paper | Reproduction | Status)
3. **Shape comparison table** (Aspect | Paper | Reproduction)
4. **Reason for difference** (single paragraph)

### Status Icons

Use consistently:
- ✅ = Match / Reproduced / Success
- ⚠️ = Partial / ~XX% / Minor difference  
- ❌ = Mismatch / Not reproduced / Failure

### Systematic Discrepancies

Name and number recurring issues:

### 1. [Name] (~magnitude)
[Description of the systematic discrepancy]

**Origin:** [Technical explanation]

### Conclusions Format

Use bold for key physics, numbered list:

> The reproduction successfully captures the **main physics** of [phenomenon]:
> 1. **Finding one** - matches paper
> 2. **Finding two** - clearly visible

End with statement on whether discrepancies affect conclusions.

---

## 11. Quick Reference

### Pre-Simulation Checklist

- [ ] Parameters extracted with sources documented
- [ ] Text/figure discrepancies resolved (figures win)
- [ ] Materials: source documented, covers wavelength range
- [ ] Geometry: spacing vs period clarified
- [ ] Resolution: adequate for physics being simulated
- [ ] Runtime: within budget

### Post-Simulation Checklist

- [ ] Results compared to paper figures
- [ ] Discrepancies classified (excellent/acceptable/investigate)
- [ ] Likely causes identified
- [ ] Data saved with clear filenames
- [ ] Plots match paper format
- [ ] Assumptions documented

### Red Flags (Stop and Investigate)

🚩 Missing expected features
🚩 Wrong trends (opposite direction)
🚩 Order of magnitude differences
🚩 Unphysical results (T > 1, A < 0)
🚩 Material validation failed

---

## 12. Future Improvements

### Multi-Model Consensus Validation

For high-confidence validation of generated simulation code, consider running the same simulation design through multiple LLMs and comparing results:

**Concept:**
- Generate simulation code with Claude
- Generate same simulation with GPT-4 and Gemini
- Compare the three implementations
- If all models produce identical or very similar code → High confidence
- If models disagree significantly → Flag for human review

**Benefits:**
- Reduces single-model bias
- Catches hallucinations and errors
- Higher confidence in correctness

**Trade-offs:**
- 3× API cost
- Increased latency
- Orchestration complexity
- Need to handle different code styles

**Implementation Notes:**
- Store generated code in structured format for comparison
- Focus on key physics (geometry, materials, BCs) not code style
- Could be a "high-confidence mode" toggle

**Not implemented in v1** — consider for future versions when base system is validated.

### Automated Figure Digitization

Currently, users must manually digitize paper figures for quantitative comparison. Future versions could:
- Use OCR to extract axis labels
- Use plot digitization tools (WebPlotDigitizer API)
- Automatically extract reference data for comparison

### Adaptive Resolution

Instead of fixed resolution estimates, future versions could:
- Start with coarse resolution
- Automatically refine based on convergence
- Stop when results stabilize

