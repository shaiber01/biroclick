# Materials Database

Optical constants for materials commonly used in optics/photonics simulations.

## Available Materials

| Material ID | Name | Type | Wavelength Range | CSV Available |
|-------------|------|------|------------------|---------------|
| `palik_silver` | Silver (Ag) | Metal | 200-1000 nm | ✅ |
| `palik_gold` | Gold (Au) | Metal | 200-2000 nm | ✅ |
| `rakic_aluminum` | Aluminum (Al) | Metal | 200-1200 nm | ✅ |
| `palik_silicon` | Silicon (Si) | Semiconductor | 300-2000 nm | ✅ |
| `malitson_sio2` | Silicon Dioxide | Dielectric | 200-2000 nm | ✅ |
| `constant_glass` | Glass (BK7-like) | Dielectric | 300-2000 nm | ❌ (n=1.51) |
| `constant_air` | Air/Vacuum | Dielectric | All | ❌ (n=1.0) |
| `constant_water` | Water | Dielectric | 300-1200 nm | ❌ (n=1.33) |

## Usage

### 1. Finding a Material

```python
import json

with open('materials/index.json') as f:
    index = json.load(f)

# Find by ID
silver = next(m for m in index['materials'] if m['material_id'] == 'palik_silver')
print(f"Silver: {silver['wavelength_range_nm']}")
```

### 2. Loading Tabulated Data

```python
import numpy as np

# Check if CSV is available
if silver['csv_available']:
    data = np.loadtxt(
        f"materials/{silver['data_file']}", 
        delimiter=',', 
        skiprows=9,  # Skip header comments
        unpack=True
    )
    wavelength_nm, n, k = data
    
    # Convert to complex refractive index
    n_complex = n + 1j * k
    
    # Convert to permittivity
    epsilon = n_complex ** 2
```

### 3. Using Drude-Lorentz Fit in Meep

```python
import meep as mp

# Get fit parameters
fit = silver['drude_lorentz_fit']

# Unit conversion: eV to Meep frequency units (where c=1, a=1µm)
# f_meep = f_eV / (hbar * c / a) = f_eV * a / (hc) 
# For a = 1 µm: f_meep ≈ f_eV * 0.8065
eV_to_meep = 1.0 / 1.23984  # More precisely: a_um / (hc in eV·µm)

susceptibilities = []

# Add Drude terms (free electrons)
for drude in fit['drude_terms']:
    susceptibilities.append(mp.DrudeSusceptibility(
        frequency=drude['omega_p_eV'] * eV_to_meep,
        gamma=drude['gamma_eV'] * eV_to_meep,
        sigma=1.0
    ))

# Add Lorentz terms (interband transitions)
for lorentz in fit['lorentz_terms']:
    susceptibilities.append(mp.LorentzianSusceptibility(
        frequency=lorentz['omega_0_eV'] * eV_to_meep,
        gamma=lorentz['gamma_eV'] * eV_to_meep,
        sigma=lorentz['sigma']
    ))

# Create Meep material
silver_material = mp.Medium(
    epsilon=fit['eps_inf'], 
    E_susceptibilities=susceptibilities
)
```

### 4. Material Validation (Stage 0)

Always validate materials before running simulations:

1. Compute ε(λ) from Drude-Lorentz fit
2. Compare to tabulated CSV data
3. Check wavelength range coverage
4. Compare to any spectra shown in paper

## Adding New Materials

### Step 1: Get Tabulated Data

Best sources:
- [refractiveindex.info](https://refractiveindex.info) - Comprehensive database
- Palik, *Handbook of Optical Constants of Solids* (1998)
- Johnson & Christy, PRB 6, 4370 (1972) - Noble metals

### Step 2: Create CSV File

Format:
```csv
# Material name and description
# Source: Citation
# Format: wavelength_nm, n (real), k (imaginary)
#
wavelength_nm,n,k
200,1.07,1.21
220,1.03,1.36
...
```

### Step 3: (Optional) Fit Drude-Lorentz Model

```python
from scipy.optimize import minimize
import numpy as np

def drude_lorentz_epsilon(omega, eps_inf, omega_p, gamma_d, lorentz_params):
    """Compute permittivity from Drude-Lorentz model."""
    # Drude term
    eps = eps_inf - omega_p**2 / (omega**2 + 1j * gamma_d * omega)
    
    # Lorentz terms
    for omega_0, gamma_l, sigma in lorentz_params:
        eps += sigma * omega_0**2 / (omega_0**2 - omega**2 - 1j * gamma_l * omega)
    
    return eps

def fit_error(params, omega, eps_data):
    """Error function for optimization."""
    # Unpack params and compute model
    eps_model = drude_lorentz_epsilon(omega, *params)
    return np.sum(np.abs(eps_model - eps_data)**2)

# Fit to your data
# result = minimize(fit_error, initial_params, args=(omega, eps_data))
```

### Step 4: Add to index.json

Follow `material_schema.json` format. Required fields:
- `material_id`: Unique snake_case identifier
- `name`: Human-readable name
- `source`: Citation
- `wavelength_range_nm`: [min, max]
- `data_format`: "wavelength_nm_n_k"
- `data_file`: Filename or null
- `csv_available`: true/false

## Data Sources

| Source | Coverage | Notes |
|--------|----------|-------|
| [refractiveindex.info](https://refractiveindex.info) | Comprehensive | Compiles multiple sources with citations |
| Palik Handbook | Comprehensive | Standard reference, 1998 |
| Rakic et al. (1998) | Metals | Drude-Lorentz fits, Applied Optics |
| Johnson & Christy (1972) | Au, Ag, Cu | PRB, widely cited |
| Malitson (1965) | SiO2 | J. Opt. Soc. Am., fused silica |

## Notes

### Metals
- **Silver**: Lowest losses, best for high-Q plasmonics
- **Gold**: Chemically stable, good biocompatibility, interband losses <500nm
- **Aluminum**: Best for UV plasmonics, forms native oxide

### Dielectrics
- **Silicon**: High-index (n~3.5), Mie resonances, absorbs above bandgap
- **SiO2**: Standard substrate (n~1.45), very low loss
- **Glass**: Use constant n=1.51 for quick estimates, SiO2 data for accuracy

### Fit Quality
- `excellent`: <5% error across fit range
- `good`: 5-15% error, suitable for most simulations
- `moderate`: 15-30% error, use tabulated data preferred
- `approximate`: Constant approximation, use for quick estimates only

