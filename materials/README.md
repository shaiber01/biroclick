# Materials Database

Comprehensive optical constants library for Meep FDTD and photonics simulations.

## Features

- **25 tabulated materials** with CSV data
- **Temperature corrections** for metals and semiconductors
- **Thin-film corrections** for nanoscale metal films
- **Uncertainty estimates** for all materials
- **Alternative fits** optimized for specific wavelength ranges
- **Validation script** for data quality checks

## Quick Start

```python
import json
import numpy as np

# Load database
with open('materials/index.json') as f:
    db = json.load(f)

# Find gold
gold = next(m for m in db['materials'] if m['material_id'] == 'palik_gold')

# Load tabulated data
data = np.loadtxt(f"materials/{gold['data_file']}", delimiter=',', skiprows=10, unpack=True)
wavelength_nm, n, k = data
```

## Available Materials

### Metals (10)
| ID | Material | λ Range | Applications |
|----|----------|---------|--------------|
| `palik_silver` | Silver | 188-1000nm | Plasmonics, SERS |
| `palik_gold` | Gold | 200-2000nm | Biosensors |
| `johnson_christy_copper` | Copper | 188-1000nm | Cheap plasmonics |
| `rakic_aluminum` | Aluminum | 200-1200nm | UV plasmonics |
| `naik_tin` | TiN | 300-2000nm | Refractory plasmonics |
| `johnson_christy_chromium` | Chromium | 200-1200nm | Adhesion layers |
| `johnson_christy_titanium` | Titanium | 200-1200nm | Adhesion layers |
| `johnson_christy_nickel` | Nickel | 200-1200nm | Magneto-optics |
| `rakic_platinum` | Platinum | 200-1200nm | Catalysis |

### Semiconductors (4)
| ID | Material | λ Range | Applications |
|----|----------|---------|--------------|
| `palik_silicon` | Silicon | 300-2000nm | Mie resonators |
| `aspnes_germanium` | Germanium | 300-2000nm | IR optics |
| `aspnes_gaas` | GaAs | 300-2000nm | Lasers, LEDs |
| `aspnes_inp` | InP | 300-2000nm | Telecom 1550nm |

### Dielectrics (9)
| ID | Material | λ Range | n (typical) |
|----|----------|---------|-------------|
| `malitson_sio2` | SiO₂ | 200-2000nm | 1.45 |
| `devore_tio2` | TiO₂ | 430-2000nm | 2.4-2.9 |
| `philipp_si3n4` | Si₃N₄ | 200-2000nm | 2.0 |
| `malitson_al2o3` | Al₂O₃ | 200-3000nm | 1.76 |
| `bright_hfo2` | HfO₂ | 200-2000nm | 1.9 |
| `bond_zno` | ZnO | 370-2000nm | 1.9-2.0 |
| `dodge_mgf2` | MgF₂ | 200-2000nm | 1.38 |
| `konig_ito` | ITO | 300-2000nm | 1.8-2.0 |
| `sultanova_pmma` | PMMA | 400-1600nm | 1.49 |

### 2D Materials (3)
| ID | Material | λ Range | Notes |
|----|----------|---------|-------|
| `kuzmenko_graphene` | Graphene | 300-2000nm | 0.34nm thickness |
| `segura_hbn` | hBN | 250-2000nm | Phonon polaritons |
| `li_mos2` | MoS₂ | 400-2000nm | 0.65nm, direct gap |

---

## Temperature Corrections

Optical constants change with temperature. For metals, the Drude damping increases:

```python
def apply_temperature_correction(material, T_kelvin):
    """Apply temperature correction to Drude parameters."""
    tc = material.get('temperature_correction', {})
    if not tc:
        return material['drude_lorentz_fit']
    
    T_ref = tc.get('reference_temperature_K', 300)
    coeff = tc.get('drude_gamma_coefficient', 0)
    
    fit = material['drude_lorentz_fit'].copy()
    
    # Scale Drude damping
    for i, drude in enumerate(fit.get('drude_terms', [])):
        gamma_ref = drude['gamma_eV']
        gamma_T = gamma_ref * (1 + coeff * (T_kelvin - T_ref))
        fit['drude_terms'][i] = {**drude, 'gamma_eV': gamma_T}
    
    return fit
```

### Temperature Correction Parameters

| Material | T_ref (K) | γ coefficient | Valid Range |
|----------|-----------|---------------|-------------|
| Silver | 300 | 0.003 | 77-600K |
| Gold | 300 | 0.0028 | 77-800K |
| Copper | 300 | 0.0035 | 77-500K |
| Aluminum | 300 | 0.004 | 77-500K |

For **semiconductors**, the bandgap shifts with temperature (Varshni equation):
```
Eg(T) = Eg(0) - α*T² / (T + β)
```

---

## Thin-Film Corrections

Bulk optical constants differ from thin films due to surface/grain boundary scattering:

```python
def apply_thin_film_correction(material, thickness_nm):
    """Apply thin-film correction to Drude parameters."""
    tfc = material.get('thin_film_correction', {})
    if not tfc or not tfc.get('bulk_data', True):
        return material['drude_lorentz_fit']
    
    critical_t = tfc.get('critical_thickness_nm', 20)
    multiplier = tfc.get('gamma_multiplier', 2.0)
    
    if thickness_nm >= critical_t:
        return material['drude_lorentz_fit']
    
    # Interpolate multiplier based on thickness
    scale = 1 + (multiplier - 1) * (1 - thickness_nm / critical_t)
    
    fit = material['drude_lorentz_fit'].copy()
    for i, drude in enumerate(fit.get('drude_terms', [])):
        fit['drude_terms'][i] = {**drude, 'gamma_eV': drude['gamma_eV'] * scale}
    
    return fit
```

### Thin-Film Parameters

| Material | Critical Thickness | γ Multiplier | Mean Free Path |
|----------|-------------------|--------------|----------------|
| Silver | 20nm | 2.5× | 52nm |
| Gold | 15nm | 2.0× | 38nm |
| Copper | 20nm | 2.2× | 40nm |
| Aluminum | 15nm | 2.5× | 16nm |

---

## Uncertainty Estimates

All materials include uncertainty estimates:

```python
material = gold
unc = material['uncertainty']

print(f"n uncertainty: ±{unc['n_uncertainty_percent']}%")
print(f"k uncertainty: ±{unc['k_uncertainty_percent']}%")

# Check for high-uncertainty regions
for region in unc.get('high_uncertainty_regions', []):
    print(f"  {region['wavelength_range_nm']}: ±{region['uncertainty_percent']}% ({region['reason']})")
```

### Typical Uncertainties

| Material Type | n Uncertainty | k Uncertainty | Notes |
|---------------|---------------|---------------|-------|
| Noble metals | 3-5% | 5-8% | Higher near interband transitions |
| Semiconductors | 1-2% | 5-10% | Well-characterized |
| Dielectrics | 0.1-3% | Large (k≈0) | n very accurate for glasses |
| 2D materials | 10-15% | 10-20% | Depends on assumed thickness |

---

## Alternative Fits

Use wavelength-specific fits for better accuracy:

```python
def get_best_fit(material, wavelength_nm):
    """Get the best Drude-Lorentz fit for a wavelength range."""
    # Check alternative fits first
    for name, alt_fit in material.get('alternative_fits', {}).items():
        wl_range = alt_fit['fit_wavelength_range_nm']
        if wl_range[0] <= wavelength_nm <= wl_range[1]:
            return alt_fit
    
    # Fall back to primary fit
    return material['drude_lorentz_fit']

# Example: Silver at 500nm
silver = next(m for m in db['materials'] if m['material_id'] == 'palik_silver')
fit = get_best_fit(silver, 500)  # Gets 'visible_only' fit
```

### Available Alternative Fits

| Material | Fit Name | Wavelength Range | Quality |
|----------|----------|------------------|---------|
| Silver | `visible_only` | 400-700nm | Excellent |
| Silver | `nir_extended` | 600-1200nm | Good |
| Gold | `nir_only` | 700-2000nm | Excellent |
| Gold | `full_visible` | 400-800nm | Good |
| Aluminum | `uv_optimized` | 200-400nm | Excellent |
| Silicon | `transparent_region` | 1100-2500nm | Excellent |

---

## Validation

Run the validation script to check data quality:

```bash
# Validate all materials
python materials/validate_materials.py

# Validate specific material
python materials/validate_materials.py palik_gold

# Generate comparison plots
python materials/validate_materials.py --plot

# Save JSON report
python materials/validate_materials.py --report
```

### Validation Checks

1. **Physical bounds**: n > 0, k ≥ 0
2. **Continuity**: No suspicious jumps in data
3. **Kramers-Kronig consistency**: Dispersion correlates with absorption
4. **Fit accuracy**: Compare tabulated data to Drude-Lorentz fit

---

## Using in Meep

```python
import meep as mp

def material_to_meep(material, wavelength_nm=None, temperature_K=300, thickness_nm=None):
    """Convert material to Meep medium."""
    
    # Get appropriate fit
    if wavelength_nm:
        fit = get_best_fit(material, wavelength_nm)
    else:
        fit = material['drude_lorentz_fit']
    
    # Apply corrections
    if temperature_K != 300:
        fit = apply_temperature_correction(material, temperature_K)
    if thickness_nm:
        fit = apply_thin_film_correction(material, thickness_nm)
    
    # Convert to Meep units (eV to Meep frequency)
    eV_to_meep = 1.0 / 1.23984  # For a = 1 µm
    
    susceptibilities = []
    
    # Drude terms
    for drude in fit.get('drude_terms', []):
        susceptibilities.append(mp.DrudeSusceptibility(
            frequency=drude['omega_p_eV'] * eV_to_meep,
            gamma=drude['gamma_eV'] * eV_to_meep,
            sigma=1.0
        ))
    
    # Lorentz terms
    for lorentz in fit.get('lorentz_terms', []):
        susceptibilities.append(mp.LorentzianSusceptibility(
            frequency=lorentz['omega_0_eV'] * eV_to_meep,
            gamma=lorentz['gamma_eV'] * eV_to_meep,
            sigma=lorentz['sigma']
        ))
    
    return mp.Medium(
        epsilon=fit.get('eps_inf', 1.0),
        E_susceptibilities=susceptibilities
    )

# Example: Gold at elevated temperature
gold = next(m for m in db['materials'] if m['material_id'] == 'palik_gold')
gold_400K = material_to_meep(gold, temperature_K=400)

# Example: Thin silver film
silver = next(m for m in db['materials'] if m['material_id'] == 'palik_silver')
silver_10nm = material_to_meep(silver, thickness_nm=10)
```

---

## Data Sources

| Source | Materials | Citation |
|--------|-----------|----------|
| Johnson & Christy (1972) | Ag, Au, Cu | PRB 6, 4370 |
| Johnson & Christy (1974) | Cr, Ti, Ni | PRB 9, 5056 |
| Rakic et al. (1998) | Al, Pt | Appl. Opt. 37, 5271 |
| Aspnes & Studna (1983) | Si, Ge, GaAs, InP | PRB 27, 985 |
| Malitson (1962, 1965) | Al₂O₃, SiO₂ | JOSA |
| refractiveindex.info | All | Compilation |

---

## Adding New Materials

1. Get data from [refractiveindex.info](https://refractiveindex.info)
2. Create CSV: `wavelength_nm,n,k`
3. Fit Drude-Lorentz model
4. Add to `index.json` following schema
5. Include uncertainty estimates
6. Add temperature/thin-film corrections if available
7. Run `validate_materials.py`

See `material_schema.json` for the full schema specification.
