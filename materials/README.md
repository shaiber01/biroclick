# Materials Database

Comprehensive optical constants library for Meep FDTD and other photonics simulations.

## Quick Stats

- **25 tabulated materials** with CSV data
- **3 constant approximations** (air, water, glass)
- Wavelength coverage: **188nm - 3000nm**
- All data from peer-reviewed sources via [refractiveindex.info](https://refractiveindex.info)

## Available Materials

### Metals (10 materials)

| Material ID | Name | λ Range | Source | Key Applications |
|-------------|------|---------|--------|------------------|
| `palik_silver` | Silver (Ag) | 188-1000nm | Johnson & Christy 1972 | Plasmonics, SERS, lowest losses |
| `palik_gold` | Gold (Au) | 200-2000nm | Palik 1998 | Biosensors, stable, biocompatible |
| `johnson_christy_copper` | Copper (Cu) | 188-1000nm | Johnson & Christy 1972 | Cheap plasmonics >600nm |
| `rakic_aluminum` | Aluminum (Al) | 200-1200nm | Rakic 1998 | UV plasmonics |
| `naik_tin` | Titanium Nitride (TiN) | 300-2000nm | Naik 2012 | **Refractory plasmonics**, CMOS |
| `johnson_christy_chromium` | Chromium (Cr) | 200-1200nm | Johnson & Christy 1974 | Adhesion layers |
| `johnson_christy_titanium` | Titanium (Ti) | 200-1200nm | Johnson & Christy 1974 | Adhesion layers |
| `johnson_christy_nickel` | Nickel (Ni) | 200-1200nm | Johnson & Christy 1974 | Magneto-optics |
| `rakic_platinum` | Platinum (Pt) | 200-1200nm | Rakic 1998 | Catalysis, high-temp |

### Semiconductors (4 materials)

| Material ID | Name | λ Range | Source | Key Applications |
|-------------|------|---------|--------|------------------|
| `palik_silicon` | Silicon (Si) | 300-2000nm | Palik 1998 | Mie resonators, n~3.5 |
| `aspnes_germanium` | Germanium (Ge) | 300-2000nm | Aspnes 1983 | IR optics, n~4-5 |
| `aspnes_gaas` | Gallium Arsenide | 300-2000nm | Aspnes 1983 | Lasers, LEDs, Eg=1.42eV |
| `aspnes_inp` | Indium Phosphide | 300-2000nm | Aspnes 1983 | **Telecom 1550nm**, PICs |

### Dielectrics (9 materials)

| Material ID | Name | λ Range | Source | Key Applications |
|-------------|------|---------|--------|------------------|
| `malitson_sio2` | Silicon Dioxide (SiO₂) | 200-2000nm | Malitson 1965 | Standard substrate, n~1.45 |
| `devore_tio2` | Titanium Dioxide (TiO₂) | 430-2000nm | Devore 1951 | **High-n metasurfaces**, n~2.4-2.9 |
| `philipp_si3n4` | Silicon Nitride (Si₃N₄) | 200-2000nm | Philipp 1973 | **Photonic waveguides**, n~2.0 |
| `malitson_al2o3` | Aluminum Oxide (Al₂O₃) | 200-3000nm | Malitson 1962 | Sapphire substrates |
| `bright_hfo2` | Hafnium Dioxide (HfO₂) | 200-2000nm | Bright 2013 | **High-k gate oxide**, n~1.9 |
| `bond_zno` | Zinc Oxide (ZnO) | 370-2000nm | Bond 1965 | TCO, piezoelectric |
| `dodge_mgf2` | Magnesium Fluoride (MgF₂) | 200-2000nm | Dodge 1984 | **AR coatings**, n~1.38 |
| `konig_ito` | Indium Tin Oxide (ITO) | 300-2000nm | König 2014 | Transparent electrodes |
| `sultanova_pmma` | PMMA (Acrylic) | 400-1600nm | Sultanova 2009 | E-beam resist, n~1.49 |

### 2D Materials (3 materials)

| Material ID | Name | λ Range | Source | Key Applications |
|-------------|------|---------|--------|------------------|
| `kuzmenko_graphene` | Graphene | 300-2000nm | Kuzmenko 2008 | Modulators, ~2.3%/layer absorption |
| `segura_hbn` | Hexagonal BN (hBN) | 250-2000nm | Segura 2018 | **Graphene encapsulation**, phonon polaritons |
| `li_mos2` | Molybdenum Disulfide (MoS₂) | 400-2000nm | Li 2014 | **Valleytronics**, TMD, direct gap |

### Constant Approximations (3 materials)

| Material ID | Name | n | Applications |
|-------------|------|---|--------------|
| `constant_glass` | Glass (BK7-like) | 1.51 | Quick estimates |
| `constant_air` | Air/Vacuum | 1.00 | Surrounding medium |
| `constant_water` | Water | 1.33 | Biosensing |

## Usage

### 1. Finding a Material

```python
import json

with open('materials/index.json') as f:
    index = json.load(f)

# Find by ID
material = next(m for m in index['materials'] if m['material_id'] == 'palik_gold')
print(f"Gold wavelength range: {material['wavelength_range_nm']}")
```

### 2. Loading Tabulated Data

```python
import numpy as np

if material['csv_available']:
    data = np.loadtxt(
        f"materials/{material['data_file']}", 
        delimiter=',', 
        skiprows=10,  # Skip header comments
        unpack=True
    )
    wavelength_nm, n, k = data
    
    # Complex refractive index
    n_complex = n + 1j * k
    
    # Permittivity
    epsilon = n_complex ** 2
```

### 3. Using in Meep

```python
import meep as mp

fit = material['drude_lorentz_fit']
eV_to_meep = 1.0 / 1.23984  # For a = 1 µm

susceptibilities = []

# Drude terms
for drude in fit['drude_terms']:
    susceptibilities.append(mp.DrudeSusceptibility(
        frequency=drude['omega_p_eV'] * eV_to_meep,
        gamma=drude['gamma_eV'] * eV_to_meep,
        sigma=1.0
    ))

# Lorentz terms
for lorentz in fit['lorentz_terms']:
    susceptibilities.append(mp.LorentzianSusceptibility(
        frequency=lorentz['omega_0_eV'] * eV_to_meep,
        gamma=lorentz['gamma_eV'] * eV_to_meep,
        sigma=lorentz['sigma']
    ))

meep_material = mp.Medium(
    epsilon=fit['eps_inf'], 
    E_susceptibilities=susceptibilities
)
```

## Material Selection Guide

### By Application

| Application | Recommended Materials |
|-------------|-----------------------|
| **Visible Plasmonics** | Ag (best Q), Au (stable), Cu (cheap) |
| **UV Plasmonics** | Al |
| **High-Temp Plasmonics** | TiN |
| **Biosensing** | Au (biocompatible) |
| **Telecom (1550nm)** | InP, Si, Ge |
| **Metasurfaces** | TiO₂, Si, Si₃N₄ |
| **Waveguides** | Si₃N₄, Si, SiO₂ |
| **AR Coatings** | MgF₂ (low-n) + TiO₂ (high-n) |
| **2D Electronics** | MoS₂, graphene, hBN |
| **Gate Dielectrics** | HfO₂ |

### By Refractive Index

| Category | Materials | Typical n |
|----------|-----------|-----------|
| **Very High** | Ge, Si, GaAs, InP | 3.0 - 5.0 |
| **High** | TiO₂, Si₃N₄, hBN, MoS₂ | 1.9 - 2.9 |
| **Medium** | SiO₂, Al₂O₃, HfO₂, ZnO, PMMA | 1.4 - 2.0 |
| **Low** | MgF₂, Air | 1.0 - 1.4 |

## Data Sources

| Reference | Materials | Citation |
|-----------|-----------|----------|
| Johnson & Christy (1972) | Ag, Au, Cu | PRB 6, 4370 |
| Johnson & Christy (1974) | Cr, Ti, Ni | PRB 9, 5056 |
| Rakic et al. (1998) | Al, Pt | Appl. Opt. 37, 5271 |
| Aspnes & Studna (1983) | Si, Ge, GaAs, InP | PRB 27, 985 |
| Malitson (1962, 1965) | Al₂O₃, SiO₂ | JOSA |
| Naik et al. (2012) | TiN | OME 2, 478 |

## Notes

### Metals
- **Ag**: Lowest losses, highest Q plasmons. Tarnishes in air.
- **Au**: Chemically stable, biocompatible. Interband losses <500nm.
- **Cu**: Good >600nm, cheap. Oxidizes in air.
- **Al**: Best for UV. Native oxide (~3nm) shifts resonances.
- **TiN**: Refractory - stable where Au/Ag melt. CMOS compatible.
- **Cr/Ti**: Adhesion layers (2-5nm) under Au/Ag/Al.

### Semiconductors
- **Si**: n~3.5, absorbs above bandgap (~1100nm). Mie resonances.
- **Ge**: n~4-5, transparent in mid-IR.
- **GaAs**: Direct gap 1.42eV, for lasers/LEDs.
- **InP**: Key for 1550nm telecom photonics.

### 2D Materials
- **Graphene**: ~2.3% absorption/layer. Use 0.34nm thickness.
- **hBN**: Insulator, encapsulates graphene. Phonon polaritons in mid-IR.
- **MoS₂**: Direct gap ~1.9eV in monolayer. Use 0.65nm thickness.

### Fit Quality
| Rating | Error | Recommendation |
|--------|-------|----------------|
| `excellent` | <5% | Use fit or tabulated |
| `good` | 5-15% | Suitable for most work |
| `moderate` | 15-30% | Prefer tabulated data |
| `approximate` | >30% | Quick estimates only |
