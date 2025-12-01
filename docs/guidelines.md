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

## 4b. Hardware Configuration

### Default Hardware Assumptions

The system assumes a **power laptop** by default:

| Resource | Default | Notes |
|----------|---------|-------|
| CPU cores | 8 | Used for Meep parallelization |
| RAM | 32 GB | Limits cell size × resolution |
| GPU | No | Future: CUDA acceleration |

### How Agents Use Hardware Config

**SimulationDesignerAgent:**
- Estimates runtime based on cell size, resolution, and cores
- Warns if memory estimate exceeds available RAM
- Suggests resolution/cell size tradeoffs

**CodeGeneratorAgent:**
- Sets appropriate number of threads for Meep
- Configures memory-efficient sweeps if needed

### Runtime Estimation Formula

```python
# Approximate runtime for 3D FDTD
cells = (cell_x * cell_y * cell_z) * resolution**3
timesteps = total_time * resolution
operations = cells * timesteps

# Empirical factor: ~1e9 operations/second/core (very rough)
runtime_seconds = operations / (1e9 * cpu_cores)
```

### Memory Estimation Formula

```python
# Meep memory usage (approximate)
bytes_per_cell = 200  # Multiple field components, double precision
total_cells = (cell_x * cell_y * cell_z) * resolution**3
memory_gb = (total_cells * bytes_per_cell) / 1e9

# Add 50% overhead for Python/numpy
memory_gb *= 1.5
```

### Overriding Default Hardware

Users can specify their hardware in the `PaperInput`:

```python
from schemas.state import HardwareConfig

custom_hardware = HardwareConfig(
    cpu_cores=16,
    ram_gb=64,
    gpu_available=False
)

paper_input = create_paper_input(
    paper_id="...",
    paper_text="...",
    hardware_config=custom_hardware
)
```

### Performance Optimization Tips

1. **Use symmetry** when possible (cuts cell size by 2×, 4×, or 8×)
2. **Start with 2D** for validation (much faster than 3D)
3. **Use subpixel smoothing** for curved geometries (more accurate at lower resolution)
4. **Reduce frequency resolution** if only peak position matters
5. **Save intermediate results** for long parameter sweeps

---

## 5. Quantitative Thresholds

### Discrepancy Classification

> **Canonical Source**: Threshold values are defined programmatically in
> `schemas/state.py:DISCREPANCY_THRESHOLDS`. The table below is for human
> reference; agents should use the state constants for consistency.

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

### Core Principle

**Validate foundations before adding complexity.** Adapt stages to what the paper actually shows.

```
Material Error ──► Propagates to ALL subsequent stages
                   (shifted resonances, wrong coupling)

Geometry Error ──► Wrong mode structure
                   (missing features, wrong Q-factors)

Numerical Error ─► Usually smaller effect
                   (noisy results, slow convergence)
```

### Always Required: Material Validation (Stage 0)

Every reproduction must validate materials first:
- Plot ε(ω), n(ω), k(ω)
- Compare to any spectra in paper
- Validate absorption peaks, linewidths
- **Requires user confirmation before proceeding**

### Paper-Dependent Stages

Include only what applies to the paper's content:

| Stage Type | When to Include | When to Skip |
|------------|-----------------|--------------|
| **Single Structure** | Paper shows isolated structures | Paper only has periodic structures |
| **Array/System** | Paper shows periodic/coupled structures | No arrays in paper |
| **Parameter Sweeps** | Paper shows multi-parameter data | Only single data points |
| **Complex Physics** | Paper involves nonlinear/transient effects | Linear steady-state only |

### Adapting to Different Paper Types

| Paper Type | Typical Staging |
|------------|-----------------|
| Material study | Materials → property analysis |
| Single nanoparticle | Materials → single structure |
| Nanoparticle array | Materials → single → array → sweeps |
| Photonic crystal | Materials → periodic structure (skip "single") |
| Waveguide | Materials → mode analysis → propagation |
| Nonlinear optics | Materials → linear validation → nonlinear |

### The Key Insight

The old "mandatory order" assumed all papers follow the same pattern (single → array → sweep). 
Real papers vary widely. The principle remains: **simpler/foundational before complex/dependent**.
But *which* stages exist depends on the paper.

---

## 7. Common Pitfalls

### Pitfall 1: Starting Too Complex

**Wrong**: "Let me simulate the full 3D periodic structure with all effects"
**Right**: "Let me validate materials first, then build up complexity appropriate to this paper"

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

## 7b. Materials Database

The system includes a materials database in `materials/` with optical constants for common materials.

### Database Structure

```
materials/
├── material_schema.json   # Schema definition for material entries
├── index.json             # Index of all available materials
├── palik_silver.csv       # Tabulated n,k data for silver
├── palik_gold.csv         # Tabulated n,k data for gold
└── rakic_aluminum.csv     # Tabulated n,k data for aluminum
```

### Using the Materials Database

**1. Finding a Material:**
```python
import json

with open('materials/index.json') as f:
    materials = json.load(f)['materials']

# Find by ID
silver = next(m for m in materials if m['material_id'] == 'palik_silver')
```

**2. Loading Tabulated Data:**
```python
import numpy as np

if silver['data_file']:
    data = np.loadtxt(f"materials/{silver['data_file']}", 
                      delimiter=',', skiprows=9)  # Skip header
    wavelength_nm, n, k = data[:, 0], data[:, 1], data[:, 2]
```

**3. Using Pre-fitted Drude-Lorentz Parameters in Meep:**
```python
import meep as mp

fit = silver['drude_lorentz_fit']
eV_to_meep = 1.0 / 1.23984  # Convert eV to Meep frequency units

susceptibilities = []

# Add Drude terms
for drude in fit['drude_terms']:
    susceptibilities.append(mp.DrudeSusceptibility(
        frequency=drude['omega_p_eV'] * eV_to_meep,
        gamma=drude['gamma_eV'] * eV_to_meep,
        sigma=1.0
    ))

# Add Lorentz terms
for lorentz in fit['lorentz_terms']:
    susceptibilities.append(mp.LorentzianSusceptibility(
        frequency=lorentz['omega_0_eV'] * eV_to_meep,
        gamma=lorentz['gamma_eV'] * eV_to_meep,
        sigma=lorentz['sigma']
    ))

material = mp.Medium(epsilon=fit['eps_inf'], E_susceptibilities=susceptibilities)
```

### Material Validation (Stage 0)

The first stage of every reproduction validates material models:

1. **Compute ε(λ)** from the Drude-Lorentz fit
2. **Compare to tabulated data** (if available)
3. **Compare to paper's material data** (if shown)
4. **User checkpoint** - confirm materials before proceeding

### Adding New Materials

To add a new material to the database:

1. Create CSV file with `wavelength_nm,n,k` columns
2. Fit Drude-Lorentz model using optimization (see `code_generator_agent.md` examples)
3. Add entry to `index.json` following `material_schema.json`
4. Document source and valid wavelength range
5. Note fit quality and any limitations

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
- [ ] Figure images extracted for vision comparison

### Figure Digitization (Deferred to V2)

> **Note**: Automatic figure digitization is planned for v2. For v1, use vision-based 
> comparison as the primary method. Digitized data can be provided manually if available.

For v1, the primary comparison method is **vision-based** (Claude's native image understanding):
- Paper figures are compared visually with simulation outputs
- Works well for qualitative shape matching, peak counting, trend detection
- Fallback to text/numerical comparison when vision is unavailable

**Optional manual digitization** (if you want quantitative metrics):

**Tool**: [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/) (free, web-based)

**Process**:
1. Load paper figure image
2. Calibrate axes (set 2 points per axis with known values)
3. Extract data points (automatic or manual)
4. Export as CSV with columns: x_value, y_value

**Naming**: `<paper_id>_<figure_id>_digitized.csv`

**Benefits of manual digitization** (optional):
- ResultsAnalyzerAgent computes MSE, correlation, R² automatically
- Removes subjective visual judgment
- Reproducible validation
- Catches small shifts that visual comparison misses

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

## 12. Meep Version Considerations

### Supported Version

This system is designed and tested with **Meep 1.28+**. Key API features used:

| Feature | API | Notes |
|---------|-----|-------|
| Flux monitors | `mp.FluxRegion()` | Syntax changed in older versions |
| Materials | `mp.Medium()` | Supports dispersive models |
| Simulation | `mp.Simulation()` | Object-oriented interface |
| Sources | `mp.Source()`, `mp.GaussianSource()` | Standard API |

### Version-Related Issues

**If resonances are shifted:**
1. Check Meep version matches expected (1.28+)
2. Verify material data interpolation is working
3. Check unit system consistency

**If code fails to run:**
1. Check for deprecated function calls
2. Verify flux region syntax matches version
3. Check material definition format

### Installation

```bash
# Recommended: conda installation
conda install -c conda-forge meep=1.28

# Verify installation
python -c "import meep; print(meep.__version__)"
```

### API Compatibility Checklist

Before running, CodeReviewerAgent verifies:
- [ ] No deprecated functions (e.g., old `get_flux_freqs()` syntax)
- [ ] Material definitions use `mp.Medium()` format
- [ ] Flux regions use current `mp.FluxRegion()` API
- [ ] Source definitions are current

---

## 14. Future Improvements

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

### Parallel Multi-Model Decision Making

Extend multi-model consensus beyond code generation to **decision-making agents**: critics, supervisors, and planners. These agents make critical choices that affect the entire workflow.

**Concept:**
```
                    ┌─────────────────┐
                    │  Same Prompt    │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
    ┌─────────┐        ┌─────────┐        ┌─────────┐
    │ Claude  │        │  GPT-4  │        │ Gemini  │
    └────┬────┘        └────┬────┘        └────┬────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  Merge/Vote     │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  Single Output  │
                    │  to Next Agent  │
                    └─────────────────┘
```

**Target Agents:**
| Agent | Decision Type | Merge Strategy |
|-------|---------------|----------------|
| **PlannerAgent** | Stage decomposition, milestones | Union of stages, intersection of critical paths |
| **SupervisorAgent** | Accept/revise/escalate decisions | Majority vote; escalate if disagreement |
| **PhysicsSanityAgent** | Validation pass/fail + concerns | Union of concerns; fail if any model fails |
| **ResultsAnalyzerAgent** | Match assessment, discrepancies | Conservative: report all discrepancies found |
| **CodeReviewerAgent** | Code approval, issues found | Union of issues; require unanimous approval |

**Merge Strategies:**

1. **Majority Vote** (for binary decisions):
   - 2/3 or 3/3 agree → Use that decision
   - Complete disagreement → Escalate to human or use most conservative

2. **Union Merge** (for lists/concerns):
   - Combine all unique items from all models
   - Example: All physics concerns from all models included

3. **Intersection Merge** (for high-confidence items):
   - Only include items all models agree on
   - Example: Only stages all models identified as critical

4. **Weighted Consensus** (for nuanced outputs):
   - Weight models by historical accuracy on similar tasks
   - Synthesize a combined response using an aggregator prompt

**Implementation Sketch:**
```python
async def parallel_agent_call(prompt: str, agent_type: str) -> MergedOutput:
    # Run same prompt on multiple models in parallel
    results = await asyncio.gather(
        call_model("claude-sonnet-4-20250514", prompt),
        call_model("gpt-4o", prompt),
        call_model("gemini-pro", prompt),
    )
    
    # Merge based on agent type
    if agent_type in ["supervisor", "reviewer"]:
        return majority_vote_merge(results)
    elif agent_type in ["planner", "analyzer"]:
        return union_merge(results)
    elif agent_type == "physics_sanity":
        return conservative_merge(results)  # Fail if ANY fails
```

**Benefits:**
- Critical decisions validated by multiple "perspectives"
- Reduces risk of single-model blind spots in planning
- Catches edge cases one model might miss
- Higher confidence in workflow decisions

**Trade-offs:**
- 3× cost for decision agents (but these are typically cheaper than code gen)
- Added latency (mitigated by parallel execution)
- Merge logic complexity
- May surface genuine ambiguity that requires human resolution

**Status**: Planned for v2. Current v1 uses single model per agent.

### Direct PDF Loading

Currently, users must convert PDFs to markdown using external tools (marker, nougat) before loading into the system. The `load_paper_from_markdown()` function then parses the markdown and downloads embedded figures.

**Future versions could handle PDFs directly:**
- Accept PDF file path as input to paper loader
- Internally convert to markdown using marker or nougat
- Extract and store figures in one step
- Handle both local PDFs and URLs to remote PDFs

**Implementation approach:**
```python
# Future API (v2)
paper_input = load_paper_from_pdf(
    pdf_path="paper.pdf",           # or URL
    output_dir="./extracted",
    converter="marker",             # or "nougat" 
    paper_id="smith2023_plasmon"
)
```

**Dependencies to add:**
- `marker-pdf` for standard PDFs
- `nougat-ocr` for complex scientific papers (equations, tables)
- Automatic fallback if one converter fails

**Current workaround:**
```bash
# Manual step (v1)
marker_single paper.pdf --output_dir ./extracted/

# Then use current loader
paper_input = load_paper_from_markdown("./extracted/paper.md", ...)
```

**Status**: Planned for v2. Current v1 requires manual PDF-to-markdown conversion.

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

### Cross-Stage Backtracking

When a significant error is discovered in a later stage that invalidates earlier work, the system can backtrack to re-run earlier stages with corrected information.

**Architecture:**
```
Any Agent (suggests backtrack) → SupervisorAgent (decides) → HANDLE_BACKTRACK → SELECT_STAGE
```

**Agents That Can Suggest Backtracking:**
- `ResultsAnalyzerAgent`: Discovers paper uses different geometry/material than assumed
- `PhysicsSanityAgent`: Physics evidence suggests fundamental setup error
- `CodeReviewerAgent`: Code review reveals design was based on wrong assumptions

**SupervisorAgent Makes Final Decision:**
- Evaluates the suggestion's validity
- Determines which stages need invalidation
- Can accept, reject, or modify the suggestion
- Enforces `MAX_BACKTRACKS` limit (default: 2)

**When to Backtrack (Critical/Significant Issues):**
| Issue Type | Example | Backtrack To |
|------------|---------|--------------|
| Wrong geometry type | Assumed spheres, paper uses rods | Stage 1 (single structure) |
| Wrong material | Used silver, paper uses gold | Stage 0 (material validation) |
| Wrong wavelength range | Simulated visible, paper is near-IR | Stage 1 |
| Wrong physics model | Used 2D, paper requires 3D | Stage 1 |

**When NOT to Backtrack (Handle Locally):**
- Minor parameter adjustments (5nm diameter difference)
- Expected discrepancies from approximations
- Numerical issues fixable in current stage
- Small optimization tweaks

**Stage Status Values for Backtracking:**
- `needs_rerun`: Backtrack target stage, will re-execute
- `invalidated`: Results invalid, will re-run after dependencies

**Limits:**
- `MAX_BACKTRACKS = 2`: Prevents infinite loops
- If limit reached + backtrack needed → escalate to user

**Status:** Planned for v1.

### Stage Parallelization

Currently, stages execute sequentially. Future versions could:
- Identify independent stages (e.g., different parameter sweep points)
- Execute independent stages in parallel
- Aggregate results before validation
- Significant speedup for parameter sweeps

**Challenges:**
- Resource management (memory, CPU)
- Error handling in parallel execution
- State synchronization
- Result aggregation

### Domain Expansion

The system is currently optimized for optics/Meep. Future expansion could include:
- **Electronics**: SPICE simulations for circuit reproduction
- **Mechanics**: FEA simulations with Fenics/Abaqus
- **Chemistry**: DFT calculations with VASP/Gaussian
- **Fluid dynamics**: CFD with OpenFOAM

Each domain would require:
- Domain-specific agent prompts
- Domain-specific validation rules
- Different simulation backends
- Domain-specific discrepancy thresholds

### Reproducibility of Reproductions

**Question**: Can the same paper be reproduced with identical results on repeated runs?

**Sources of variation:**
- LLM non-determinism (even with temperature=0)
- Meep numerical noise
- Random initialization in some simulations

**Potential solutions:**
- Seed fixing for deterministic LLM outputs (when available)
- Seed fixing for numerical simulations
- Tolerance-based comparison for "identical" results
- Logging all random seeds used

**Status**: Documented for future investigation.

### Platform Support and Compatibility

ReproLab is designed to run on Unix-like systems (Linux, macOS) and has limited
Windows support in v1. This section documents platform-specific considerations.

#### Supported Platforms

| Platform | Support Level | Notes |
|----------|--------------|-------|
| **Linux (x64)** | ✅ Full | Primary development platform |
| **macOS (Intel)** | ✅ Full | Tested on macOS 12+ |
| **macOS (Apple Silicon)** | ✅ Full | M1/M2/M3 via Rosetta or native |
| **Windows 10/11** | ⚠️ Limited | See Windows limitations below |
| **WSL2** | ✅ Full | Recommended for Windows users |

#### Windows Limitations (v1)

Windows support is limited due to the following platform differences:

**1. Memory Limiting Not Available**

The `resource` module used for memory limits is Unix-only:
```python
# This works on Unix, fails on Windows
import resource
resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
```

**Impact**: Simulations can consume unlimited memory on Windows, potentially causing system instability.

**Workaround**: The system automatically skips memory limits on Windows. Monitor task manager manually.

**2. Process Group Signaling Differences**

Unix uses `os.killpg()` for killing process groups; Windows requires different APIs.

**Impact**: Timeout enforcement may not kill all child processes on Windows.

**Workaround**: The system uses `subprocess.Popen.terminate()` on Windows, which handles most cases.

**3. `preexec_fn` Not Supported**

The `preexec_fn` parameter to `subprocess.run()` doesn't work on Windows:
```python
# This silently does nothing on Windows
subprocess.run(..., preexec_fn=set_limits if os.name != 'nt' else None)
```

**Impact**: Cannot set resource limits before simulation starts.

**4. Path Separator Differences**

Unix uses `/`, Windows uses `\`. While Python's `pathlib` handles this, some edge cases exist.

**Impact**: File paths in generated code may need adjustment.

**Workaround**: Always use `pathlib.Path` instead of string concatenation.

#### Windows Workarounds and Alternatives

**Option 1: Use WSL2 (Recommended)**

Windows Subsystem for Linux 2 provides a full Linux environment:

```powershell
# Install WSL2 (PowerShell as Administrator)
wsl --install -d Ubuntu-22.04

# Install Miniconda in WSL
wsl
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Install Meep
conda create -n reprolab python=3.11
conda activate reprolab
conda install -c conda-forge meep=1.28.0

# Clone and setup ReproLab
git clone https://github.com/youruser/reprolab.git
cd reprolab
pip install -r requirements.txt
```

WSL2 advantages:
- Full Unix support including resource limits
- Native Meep installation via conda
- Access to Windows files via `/mnt/c/`
- VS Code WSL integration works seamlessly

**Option 2: Docker Desktop for Windows**

Run simulations in a Linux container:

```powershell
# Pull or build the Meep container
docker pull condaforge/mambaforge

# Run with resource limits (works on Windows!)
docker run --rm -it `
    --memory=8g `
    --cpus=4 `
    -v ${PWD}/outputs:/outputs `
    condaforge/mambaforge bash

# Inside container: install and run
conda install -c conda-forge meep=1.28.0
pip install -r requirements.txt
python -m src.graph ...
```

Docker advantages:
- Full resource limiting works via Docker's runtime
- Consistent environment across platforms
- Isolation from host system

**Option 3: Native Windows (Limited)**

If you must run natively on Windows:

1. Install Meep via conda (may have issues):
   ```cmd
   conda install -c conda-forge meep
   ```

2. Set `REPROLAB_SKIP_RESOURCE_LIMITS=1` to suppress warnings

3. Monitor simulations manually - they can consume unlimited resources

4. Consider shorter timeout values to catch runaway simulations

**Native Windows Limitations Summary:**
| Feature | Unix | Windows Native |
|---------|------|----------------|
| Timeout enforcement | ✅ | ⚠️ Basic only |
| Memory limits | ✅ | ❌ Not available |
| CPU core limits | ✅ | ✅ Via env vars |
| Process isolation | ✅ | ⚠️ Limited |
| Meep stability | ✅ | ⚠️ May vary |

#### Code Runner Platform Detection

The code runner automatically adapts to the platform:

```python
import os
import sys

def get_platform_config():
    """Detect platform and return appropriate configuration."""
    is_windows = sys.platform == 'win32'
    is_wsl = 'microsoft' in os.uname().release.lower() if hasattr(os, 'uname') else False
    
    if is_windows:
        return {
            'memory_limiting_available': False,
            'process_group_kill_available': False,
            'preexec_fn_available': False,
            'recommended_action': 'Consider using WSL2 or Docker for full functionality',
        }
    elif is_wsl:
        return {
            'memory_limiting_available': True,  # WSL2 supports resource module
            'process_group_kill_available': True,
            'preexec_fn_available': True,
            'recommended_action': None,
        }
    else:  # Linux, macOS
        return {
            'memory_limiting_available': True,
            'process_group_kill_available': True,
            'preexec_fn_available': True,
            'recommended_action': None,
        }
```

#### Environment Variables for Platform Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `REPROLAB_SKIP_RESOURCE_LIMITS` | `0` | Set to `1` to skip memory limiting (auto-set on Windows) |
| `REPROLAB_TIMEOUT_BUFFER_SECONDS` | `30` | Extra time before hard kill |
| `REPROLAB_USE_PROCESS_GROUP` | `1` | Set to `0` to disable process group kills |

#### Meep Installation by Platform

**Linux (Ubuntu/Debian)**
```bash
conda install -c conda-forge meep=1.28.0
# Or with MPI support:
conda install -c conda-forge meep=1.28.0 mpi4py openmpi
```

**macOS (Intel)**
```bash
conda install -c conda-forge meep=1.28.0
```

**macOS (Apple Silicon M1/M2/M3)**
```bash
# Native ARM64 build (recommended)
CONDA_SUBDIR=osx-arm64 conda install -c conda-forge meep=1.28.0

# Or via Rosetta (if ARM build has issues)
arch -x86_64 conda install -c conda-forge meep=1.28.0
```

**Windows Native (Limited Support)**
```cmd
:: May have stability issues
conda install -c conda-forge meep
```

**WSL2 (Recommended for Windows)**
```bash
# Inside WSL2 Ubuntu
conda install -c conda-forge meep=1.28.0
```

**Verify Installation**
```python
import meep as mp
print(f"Meep version: {mp.__version__}")
print(f"Number of processes: {mp.count_processors()}")
```

### Sandboxed Code Execution

LLM-generated Meep code must be executed safely to prevent:
- Runaway simulations consuming all resources
- Malicious or buggy code affecting the host system
- Infinite loops blocking the workflow

#### V1 Implementation: Subprocess with Resource Limits

For the initial version, we use Python subprocess with timeout and memory limits.
See `src/code_runner.py` for the full implementation.

**Core approach:**
```python
import subprocess
import resource
import os
import time
from pathlib import Path

def run_simulation_sandboxed(
    code: str,
    stage_id: str,
    output_dir: Path,
    timeout_seconds: int = 3600,
    max_memory_gb: float = 8.0,
    max_cpu_cores: int = 4
) -> dict:
    """Execute Meep simulation in sandboxed subprocess."""
    
    # Write code to file
    script_path = output_dir / f"simulation_{stage_id}.py"
    script_path.write_text(code)
    
    # Resource limits (Unix only)
    def set_limits():
        max_bytes = int(max_memory_gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
    
    # Execute with controlled environment
    result = subprocess.run(
        ["python", str(script_path)],
        cwd=str(output_dir),
        timeout=timeout_seconds,
        capture_output=True,
        text=True,
        preexec_fn=set_limits if os.name != 'nt' else None,
        env={
            **os.environ,
            "OMP_NUM_THREADS": str(max_cpu_cores),
            "OPENBLAS_NUM_THREADS": str(max_cpu_cores),
            "MKL_NUM_THREADS": str(max_cpu_cores),
        }
    )
    
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.returncode,
        "output_files": [f.name for f in output_dir.glob("*") if f.is_file()],
    }
```

**Key features:**
- **Timeout**: `subprocess.run(timeout=...)` kills process after time limit
- **Memory limit**: `resource.setrlimit(RLIMIT_AS, ...)` sets address space limit
- **Thread limits**: Environment variables control OpenMP/BLAS parallelism
- **Working directory isolation**: Each stage runs in its own output directory
- **Output capture**: stdout/stderr captured for debugging

**V1 Limitations:**
| Limitation | Risk Level | Mitigation |
|------------|------------|------------|
| No network isolation | Medium | Simulations shouldn't need network |
| Filesystem access | Medium | Working directory is isolated |
| Memory limits are soft (Unix) | Low | Process killed if exceeded |
| Windows compatibility | Medium | Use job objects on Windows |

#### Future: Docker Container (v2)

For production deployments, Docker provides better isolation:

```bash
docker run --rm \
    --memory=8g \
    --cpus=4 \
    --network=none \
    --read-only \
    -v $(pwd)/outputs:/outputs:rw \
    meep-sandbox:latest \
    python simulation.py
```

**Docker advantages:**
- Network isolation (`--network=none`)
- Read-only filesystem (`--read-only`)
- Hard memory limits
- User namespace isolation
- Reproducible environment

**Status**: V1 uses subprocess (implemented in `src/code_runner.py`). Docker support planned for v2.

---

## 15. Self-Improving System Roadmap

### Current: PromptAdaptorAgent (v1)

The system includes a `PromptAdaptorAgent` that customizes agent prompts for each paper:
- Runs BEFORE all other agents
- Analyzes paper domain, materials, techniques
- Adds, modifies, or disables prompt content
- Documents all changes with reasoning
- Saves adaptation log for review

**Current capabilities:**
- Append domain-specific guidance to prompts
- Modify existing content (with high confidence)
- Disable irrelevant content (with very high confidence)
- Document all changes

**Constraints:**
- Cannot modify `global_rules.md`
- Cannot modify workflow structure
- Cannot add/remove agents

### Planned: PromptEvolutionAgent (v2)

A meta-agent that learns from accumulated prompt adaptations:

**Concept:**
```
Paper 1 → PromptAdaptor → Adaptations logged
Paper 2 → PromptAdaptor → Adaptations logged  
Paper 3 → PromptAdaptor → Adaptations logged
         ↓
PromptEvolutionAgent (periodic review)
         ↓
"These adaptations appeared in 80% of plasmonics papers:
 - J-aggregate fitting procedure
 - Rabi splitting thresholds
 Recommend adding to base SimulationDesignerAgent."
         ↓
Human review → Merge to base prompts
```

**Benefits:**
- System improves over time
- Domain-specific knowledge accumulates
- Reduces per-paper adaptation cost
- Captures best practices automatically

**Implementation Notes:**
- Run periodically (every N papers or manually triggered)
- Analyze adaptation logs across papers
- Identify high-frequency, high-impact additions
- Generate recommendations with confidence scores
- Human approval required before base prompt changes

### Planned: Dynamic Workflow Adaptation (v3)

Extend PromptAdaptorAgent to modify workflow structure:

**Potential Capabilities:**
- Add specialized validation nodes for specific paper types
- Skip irrelevant nodes for simple papers
- Add parallel execution paths for independent stages
- Insert domain-specific agents dynamically

**Examples:**
```python
# Simple paper (transmission spectrum only)
PLAN → DESIGN → CODE → RUN → ANALYZE → SUPERVISOR

# Complex paper (strong coupling + near-field)
PLAN → DESIGN → CODE → RUN → NEARFIELD_ANALYZE → COUPLING_VALIDATE → ANALYZE → SUPERVISOR
```

**Challenges:**
- Workflow validation (ensure graph is acyclic, complete)
- State management across dynamic nodes
- Testing dynamic configurations
- Debugging failures in modified workflows

**Not implemented in v1** — requires stable base system first.

### System Evolution Path

```
v1.0 (Current):
├── 10 fixed agents
├── PromptAdaptorAgent customizes prompts
└── All adaptations logged

v2.0 (Future):
├── PromptEvolutionAgent learns from logs
├── Base prompts improve over time
└── Reduced per-paper adaptation

v3.0 (Future):
├── Dynamic workflow adaptation
├── Domain-specific agent injection
└── Fully adaptive reproduction system
```

### Adaptation Logging Schema

All adaptations are logged for future learning. The formal schema is defined in
`schemas/prompt_adaptations_schema.json`.

**Key fields:**
- `paper_id`, `timestamp`, `domain`: Identification
- `domain_signals`: Keywords that detected the domain
- `adaptations[]`: List of modifications with type, confidence, content, reason
- `reproduction_outcome`: Post-completion success metrics
- `adaptation_effectiveness`: Retrospective evaluation per adaptation

**Example (abbreviated):**
```json
{
  "paper_id": "paper_xyz",
  "timestamp": "2025-11-30T12:00:00Z",
  "domain": "plasmonics_strong_coupling",
  "adaptations": [
    {
      "id": "APPEND_001",
      "target_agent": "SimulationDesignerAgent",
      "modification_type": "append",
      "confidence": 0.85,
      "content": "For J-aggregate materials: Use Lorentzian oscillator model...",
      "reason": "Paper involves J-aggregate excitons"
    }
  ],
  "adaptation_effectiveness": {
    "APPEND_001": "helpful"
  }
}
```

See `schemas/prompt_adaptations_schema.json` for the complete schema with all fields,
validation rules, and detailed examples. This schema supports future machine learning
on adaptation effectiveness.

---

## 15b. API Cost Estimation

Understanding token usage helps estimate costs before running reproductions.

### Token Usage by Agent

| Agent | Typical Input Tokens | Typical Output Tokens | Notes |
|-------|---------------------|----------------------|-------|
| **PromptAdaptorAgent** | 15K-25K | 1K-2K | Scans paper text (truncated) |
| **PlannerAgent** | 30K-50K | 3K-5K | Full paper + figures, largest input |
| **SimulationDesignerAgent** | 5K-10K | 1K-3K | Stage context + assumptions |
| **CodeGeneratorAgent** | 3K-5K | 2K-4K | Design spec + feedback |
| **CodeReviewerAgent** | 4K-8K | 1K-2K | Design or code to review |
| **ExecutionValidatorAgent** | 2K-4K | 0.5K-1K | Simulation outputs |
| **PhysicsSanityAgent** | 3K-6K | 1K-2K | Output data analysis |
| **ResultsAnalyzerAgent** | 10K-20K | 2K-4K | Includes figure images |
| **ComparisonValidatorAgent** | 5K-10K | 1K-2K | Comparison validation |
| **SupervisorAgent** | 8K-15K | 1K-3K | Summary context |

### Typical Reproduction Costs

| Paper Complexity | Stages | Agent Calls | Est. Input Tokens | Est. Output Tokens | Est. Cost (Opus) |
|-----------------|--------|-------------|-------------------|--------------------|--------------------|
| Simple (1-2 figures) | 2-3 | 15-25 | 200K-400K | 30K-60K | $3-8 |
| Medium (3-5 figures) | 4-5 | 30-50 | 500K-800K | 80K-120K | $10-20 |
| Complex (6+ figures) | 6-8 | 60-100 | 1M-1.5M | 150K-250K | $25-50 |

*Cost estimates based on Claude Opus 4.5 pricing (~$15/M input, ~$75/M output). Actual costs vary.*

### Cost Breakdown by Workflow Phase

| Phase | % of Total Cost | Notes |
|-------|-----------------|-------|
| Planning (Planner + Adaptor) | 25-35% | Large paper input |
| Design/Code Generation | 20-30% | Multiple revision cycles |
| Validation (Reviewers) | 15-25% | Per-stage reviews |
| Analysis (Analyzer + Comparison) | 20-30% | Figure images add tokens |
| Supervision | 5-10% | Summary-level context |

### Cost Optimization Strategies

1. **Trim Paper Text Before Loading**
   - Remove references section (20-30% savings)
   - Remove acknowledgments, author contributions
   - Keep Methods and Results sections intact
   
2. **Manual Figure Digitization (Optional)**
   - Digitized CSV data enables quantitative metrics (MSE, R²)
   - Vision comparison is primary method for v1
   - Auto-digitization planned for v2
   
3. **Use Debug Mode First**
   - Quick validation run catches obvious issues
   - Prevents wasted tokens on fundamentally broken setups
   
4. **Reduce Revision Cycles**
   - Well-prepared paper inputs reduce planning revisions
   - Good material data reduces Stage 0 iterations
   
5. **Future: Use Cheaper Models for Validators**
   - v2 will support per-agent model selection
   - Validation agents can use Claude Sonnet (5x cheaper)

### Token Counting

The system tracks token usage in the metrics log:

```python
# Access token metrics after a run
metrics = state["metrics"]
print(f"Total input tokens: {metrics['total_input_tokens']:,}")
print(f"Total output tokens: {metrics['total_output_tokens']:,}")

# Per-agent breakdown
for call in metrics["agent_calls"]:
    print(f"{call['agent']}: {call.get('input_tokens', 0):,} in / {call.get('output_tokens', 0):,} out")
```

### Cost Estimation Before Running

```python
from src.paper_loader import estimate_token_cost

# Before running, estimate costs
cost_estimate = estimate_token_cost(paper_input)
print(f"Paper text: ~{cost_estimate['text_tokens']:,} tokens")
print(f"Figures: ~{cost_estimate['figure_tokens']:,} tokens")
print(f"Estimated total cost: ${cost_estimate['estimated_cost_usd']:.2f}")

if cost_estimate['estimated_cost_usd'] > 20:
    print("⚠️ High cost - consider trimming paper or using debug mode first")
```

---

## 15c. Structured Output with Function Calling

### The Problem with Free-Text JSON

When asking LLMs to output JSON directly in their response, several issues can occur:
- Extra text before/after the JSON ("Here's the output:", etc.)
- Markdown code block wrapping (```json ... ```)
- JSON syntax errors (trailing commas, unquoted keys)
- Missing required fields
- Type mismatches

### Recommended: Function Calling APIs

Use OpenAI's function calling or Anthropic's tool use APIs for structured outputs.
This guarantees schema compliance and eliminates parsing errors.

**OpenAI Example:**

```python
from openai import OpenAI
import json

client = OpenAI()

# Load our JSON schema
with open("schemas/plan_schema.json") as f:
    plan_schema = json.load(f)

# Define tool with our schema
tools = [{
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "Submit the reproduction plan",
        "parameters": plan_schema  # Use schema directly!
    }
}]

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": planner_agent_prompt},
        {"role": "user", "content": paper_text}
    ],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "submit_plan"}}
)

# Guaranteed valid JSON matching schema
plan = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
```

**Anthropic Example:**

```python
import anthropic
import json

client = anthropic.Anthropic()

# Define tool with our schema
tools = [{
    "name": "submit_plan",
    "description": "Submit the reproduction plan",
    "input_schema": plan_schema  # Use schema directly!
}]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    system=planner_agent_prompt,
    messages=[{"role": "user", "content": paper_text}],
    tools=tools,
    tool_choice={"type": "tool", "name": "submit_plan"}
)

# Guaranteed valid JSON matching schema
plan = response.content[0].input
```

### Benefits

| Aspect | Free-text JSON | Function Calling |
|--------|---------------|------------------|
| Schema compliance | Manual validation needed | Guaranteed by API |
| Parsing errors | Common | Impossible |
| Type safety | Runtime errors | Compile-time safety |
| Required fields | May be missing | Always present |
| Debugging | Parse error messages | Clear validation errors |

### Implementation Pattern for Agents

Each agent should:
1. Define a tool matching its output schema
2. Force tool use with `tool_choice`
3. Extract result from tool call response
4. No JSON parsing needed—API handles it

```python
def call_agent(agent_name: str, system_prompt: str, user_input: str, output_schema: dict) -> dict:
    """Generic agent caller using function calling."""
    tool = {
        "type": "function",
        "function": {
            "name": f"submit_{agent_name}_output",
            "description": f"Submit {agent_name} output",
            "parameters": output_schema
        }
    }
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": f"submit_{agent_name}_output"}}
    )
    
    return json.loads(response.choices[0].message.tool_calls[0].function.arguments)
```

### Schema-First Design Rule

**Rule**: All agents with structured outputs MUST have a JSON schema in `schemas/`.
Tool definitions for function calling are constructed directly from these schemas,
never hand-specified in code.

This prevents schema drift between:
- JSON validation schemas (`schemas/*.json`)
- Python TypedDicts (generated from schemas via `datamodel-codegen`)
- LLM function calling tool definitions

**When adding a new structured output:**

1. Define the schema in `schemas/<name>_schema.json`
2. Generate Python types: `datamodel-codegen --input schemas/<name>_schema.json ...`
3. Build tool definitions by loading the JSON schema directly (as shown in examples above)

**Anti-pattern (don't do this):**

```python
# BAD: Hand-written tool schema that will drift from JSON schema
tools = [{
    "type": "function",
    "function": {
        "name": "submit_plan",
        "parameters": {
            "type": "object",
            "properties": {
                "stages": {"type": "array"},  # Schema drift risk!
                # ... manually maintained, will diverge from plan_schema.json
            }
        }
    }
}]
```

**Correct pattern:**

```python
# GOOD: Load schema from single source of truth
with open("schemas/plan_schema.json") as f:
    plan_schema = json.load(f)

tools = [{
    "type": "function",
    "function": {
        "name": "submit_plan",
        "parameters": plan_schema  # Always in sync
    }
}]
```

### Fallback for Non-Compliant Models

If using a model without function calling support:

```python
import re
import json

def extract_json_from_response(response_text: str) -> dict:
    """Fallback JSON extraction with cleanup."""
    # Remove markdown code blocks
    cleaned = re.sub(r'```json?\s*', '', response_text)
    cleaned = re.sub(r'```\s*$', '', cleaned)
    
    # Find JSON object
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if not match:
        raise ValueError("No JSON object found in response")
    
    # Parse and validate
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        # Try fixing common issues
        fixed = match.group()
        fixed = re.sub(r',\s*}', '}', fixed)  # Remove trailing commas
        fixed = re.sub(r',\s*]', ']', fixed)
        return json.loads(fixed)
```

**Recommendation:** Always prefer function calling. Only use fallback for models that
don't support it or during development/testing.

