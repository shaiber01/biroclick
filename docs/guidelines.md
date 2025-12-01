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
- [ ] **Key figures digitized** (strongly recommended for quantitative comparison)

### Figure Digitization (Recommended)

For quantitative validation, digitize key figures BEFORE starting reproduction:

**Tool**: [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/) (free, web-based)

**Process**:
1. Load paper figure image
2. Calibrate axes (set 2 points per axis with known values)
3. Extract data points (automatic or manual)
4. Export as CSV with columns: x_value, y_value

**Naming**: `<paper_id>_<figure_id>_digitized.csv`

**Benefits**:
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

## 15. Structured Output with Function Calling

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

