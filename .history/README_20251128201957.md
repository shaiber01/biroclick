# Eizner et al. 2015 Paper Reproduction

Computational reproduction of **"Aluminum Nanoantenna Complexes for Strong Coupling between Excitons and Localized Surface Plasmons"** (Nano Letters 2015, 15, 6215-6221).

## Summary

This project reproduces the FDTD simulation results from the paper using **Python + Meep**. The main findings about strong coupling between TDBC J-aggregate excitons and aluminum nanoantenna LSPs are successfully reproduced.

### Key Results
- ✅ **Rabi splitting: ~0.4 eV** (matches paper)
- ✅ **Anti-crossing** in transmission dispersion
- ✅ **Polarization-dependent** nanorod response
- ⚠️ **Field enhancement: ~50%** of paper's values
- ⚠️ **LSP positions: ~50-100nm redshift** vs paper

## Files

### Reproduction Scripts
| File | Figure | Description |
|------|--------|-------------|
| `fig2a_exact.py` | 2a | TDBC absorption/emission spectra |
| `fig2bc_corrected.py` | 2b,c | Electric field enhancement maps |
| `fig3_fast.py` | 3c,d | Nanodisk transmission maps |
| `fig3ef_exact.py` | 3e,f | Coupled oscillator dispersion |
| `fig4_nanorods.py` | 4 | Nanorod transmission spectra |
| `fig5_proper.py` | 5 | Emission enhancement |

### Supporting Files
| File | Description |
|------|-------------|
| `palik_aluminum.py` | Palik Al Drude-Lorentz model |
| `REPRODUCTION_REPORT.md` | Detailed figure-by-figure analysis |
| `article.pdf` | Original paper |

## Requirements

```bash
conda activate meep-optics  # or your Meep environment
```

- Python 3.x
- Meep (FDTD)
- NumPy
- Matplotlib
- SciPy

## Running

```bash
# Quick analytical figures
python fig2a_exact.py
python fig3ef_exact.py

# FDTD simulations (slower)
python fig2bc_corrected.py   # ~5 min
python fig3_fast.py          # ~10 min
python fig4_nanorods.py      # ~15 min
python fig5_proper.py        # ~5 min
```

## Critical Corrections Applied

1. **TDBC linewidth:** Paper's Methods gives γ that's 4× too narrow; corrected using Figure 2a
2. **Nanorod shape:** Paper says "modeled as ellipses" - changed from Block to Ellipsoid
3. **Conformal TDBC coating:** TDBC wraps around nanostructure, not just flat on top
4. **Emission sample:** Used reduced Rabi splitting (0.25 eV) for Figure 5

## See Also

See `REPRODUCTION_REPORT.md` for detailed analysis of what matches and doesn't match the paper.
