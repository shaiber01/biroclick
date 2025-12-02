# Materials Database

Optical constants for materials commonly used in optics/photonics simulations.

## Available Materials

### Metals

| Material ID | Name | Wavelength Range | Source | Applications |
|-------------|------|------------------|--------|--------------|
| `palik_silver` | Silver (Ag) | 188-1000 nm | Johnson & Christy 1972 | Plasmonics, SERS, metamaterials |
| `palik_gold` | Gold (Au) | 200-2000 nm | Palik 1998 | Biosensors, nanomedicine |
| `johnson_christy_copper` | Copper (Cu) | 188-1000 nm | Johnson & Christy 1972 | Plasmonics, interconnects |
| `rakic_aluminum` | Aluminum (Al) | 200-1200 nm | Rakic 1998 | UV plasmonics |
| `johnson_christy_chromium` | Chromium (Cr) | 200-1200 nm | Johnson & Christy 1974 | Adhesion layers |
| `johnson_christy_titanium` | Titanium (Ti) | 200-1200 nm | Johnson & Christy 1974 | Adhesion layers |
| `johnson_christy_nickel` | Nickel (Ni) | 200-1200 nm | Johnson & Christy 1974 | Magneto-optics |
| `rakic_platinum` | Platinum (Pt) | 200-1200 nm | Rakic 1998 | Catalysis, electrochemistry |

### Semiconductors

| Material ID | Name | Wavelength Range | Source | Applications |
|-------------|------|------------------|--------|--------------|
| `palik_silicon` | Silicon (Si) | 300-2000 nm | Palik 1998 | Mie resonators, waveguides |
| `aspnes_germanium` | Germanium (Ge) | 300-2000 nm | Aspnes 1983 | IR optics, high-n resonators |
| `aspnes_gaas` | Gallium Arsenide (GaAs) | 300-2000 nm | Aspnes 1983 | Lasers, LEDs, quantum dots |

### Dielectrics

| Material ID | Name | Wavelength Range | Source | Applications |
|-------------|------|------------------|--------|--------------|
| `malitson_sio2` | Silicon Dioxide (SiO₂) | 200-2000 nm | Malitson 1965 | Substrates, spacers |
| `devore_tio2` | Titanium Dioxide (TiO₂) | 430-2000 nm | Devore 1951 | Metasurfaces, photocatalysis |
| `philipp_si3n4` | Silicon Nitride (Si₃N₄) | 200-2000 nm | Philipp 1973 | Waveguides, integrated photonics |
| `malitson_al2o3` | Aluminum Oxide (Al₂O₃) | 200-3000 nm | Malitson 1962 | Substrates, protective coatings |
| `konig_ito` | Indium Tin Oxide (ITO) | 300-2000 nm | König 2014 | Transparent electrodes |
| `constant_glass` | Glass (BK7-like) | 300-2000 nm | Constant n=1.51 | Quick estimates |
| `constant_air` | Air/Vacuum | All | Constant n=1.0 | Surrounding medium |
| `constant_water` | Water | 300-1200 nm | Constant n=1.33 | Biosensing |

### 2D Materials

| Material ID | Name | Wavelength Range | Source | Applications |
|-------------|------|------------------|--------|--------------|
| `kuzmenko_graphene` | Graphene | 300-2000 nm | Kuzmenko 2008 | Modulators, photodetectors |

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
        skiprows=10,  # Skip header comments
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
| Palik Handbook (1998) | Comprehensive | Standard reference |
| Rakic et al. (1998) | Metals | Drude-Lorentz fits, Applied Optics |
| Johnson & Christy (1972, 1974) | Metals | PRB, widely cited |
| Aspnes & Studna (1983) | Semiconductors | PRB, Si, Ge, GaAs |
| Malitson (1962, 1965) | Oxides | J. Opt. Soc. Am., SiO₂, Al₂O₃ |

## Notes

### Metals
- **Silver**: Lowest losses, best for high-Q plasmonics. Tarnishes in air.
- **Gold**: Chemically stable, good biocompatibility, interband losses <500nm
- **Copper**: Good plasmonics >600nm, cheap, oxidizes in air
- **Aluminum**: Best for UV plasmonics, forms native oxide (~2-3nm)
- **Chromium/Titanium**: Adhesion layers (2-5nm) for Au/Ag/Al films
- **Nickel**: Ferromagnetic, enables magneto-optical control
- **Platinum**: Catalytic, chemically inert, high-temperature stable

### Semiconductors
- **Silicon**: High-index (n~3.5), Mie resonances, absorbs above bandgap (~1100nm)
- **Germanium**: Very high-index (n~4), transparent in mid-IR
- **GaAs**: Direct bandgap (~870nm), for lasers and LEDs

### Dielectrics
- **SiO₂**: Standard substrate (n~1.45), very low loss
- **TiO₂**: High-index (n~2.4), photocatalysis, metasurfaces
- **Si₃N₄**: Waveguide material (n~2.0), integrated photonics
- **Al₂O₃**: Sapphire substrates, native oxide on Al
- **ITO**: Transparent conductor, NIR plasmonics

### 2D Materials
- **Graphene**: ~2.3% absorption per layer, effective constants assume 0.34nm thickness

### Fit Quality
- `excellent`: <5% error across fit range
- `good`: 5-15% error, suitable for most simulations
- `moderate`: 15-30% error, prefer tabulated data
- `approximate`: Constant approximation, quick estimates only
