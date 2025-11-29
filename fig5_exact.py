#!/usr/bin/env python3
"""
Figure 5 - EXACT paper format
Emission enhancement

Paper specifications:
- (a) 2D emission map:
  - X-axis: Diameter (nm) 75-205 (labels at 85, 105, 125, 145, 165, 195)
  - Y-axis: Wavelength (nm) 500-650 (inverted)
  - Colormap: 1.2-2.0 enhancement (hot/orange colormap)
  - Yellow dashed line: pump wavelength 530nm

- (b-f) Individual spectra for D=105,125,140,155,205nm:
  - X-axis: Wavelength (nm) 400-700
  - Left Y-axis: Emission Enhancement 1.0-2.0 (black solid line)
  - Right Y-axis: 1-T_norm 0-0.6 (blue dashed line)

KEY: Emission sample had REDUCED Rabi splitting (0.25 eV)
"""

import meep as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

print("=" * 60)
print("FIGURE 5 - Emission Enhancement (Paper Format)")
print("=" * 60)

# Palik Al
Al_Palik = mp.Medium(
    epsilon=1.0,
    E_susceptibilities=[
        mp.DrudeSusceptibility(frequency=12.10, gamma=0.081, sigma=1.0),
        mp.LorentzianSusceptibility(frequency=1.21, gamma=0.40, sigma=2.0)
    ]
)

# Materials
glass = mp.Medium(epsilon=1.51**2)
omega_p_ITO = 1.78e15 / (2 * np.pi * 3e14)
gamma_ITO = 1.5e14 / (2 * np.pi * 3e14)
ITO = mp.Medium(epsilon=3.9, E_susceptibilities=[
    mp.DrudeSusceptibility(frequency=omega_p_ITO, gamma=gamma_ITO, sigma=1.0)
])

# TDBC with REDUCED oscillator strength (Rabi = 0.25 eV for emission sample)
omega_X = 3.22e15 / (2 * np.pi * 3e14)
gamma_X = 1.0e14 / (2 * np.pi * 3e14)
f_TDBC = 0.15  # Reduced for 0.25 eV splitting
TDBC = mp.Medium(epsilon=2.56, E_susceptibilities=[
    mp.LorentzianSusceptibility(frequency=omega_X, gamma=gamma_X, sigma=f_TDBC)
])

# Parameters
diameters_map = np.array([75, 95, 115, 140, 155, 185, 205])  # For 2D map
diameters_specific = np.array([105, 125, 140, 155, 205])      # For panels b-f

h_disk = 0.040
h_ITO = 0.030
h_TDBC = 0.020
gap = 0.180

resolution = 60
sz = 2.0
dpml = 0.4

wl_min, wl_max = 0.45, 0.70
freq_min, freq_max = 1/wl_max, 1/wl_min
fcen = (freq_min + freq_max) / 2
df = freq_max - freq_min
nfreq = 100

def simulate_T_norm(D_nm):
    """Simulate T_disk / T_tdbc (normalized transmission)."""
    D = D_nm / 1000
    period = D + gap
    sx = sy = period
    cell_size = mp.Vector3(sx, sy, sz)
    
    z_bottom = -sz/2 + dpml
    z_glass_top = z_bottom + 0.5
    z_ITO_top = z_glass_top + h_ITO
    z_source = z_glass_top - 0.15
    z_trans = z_ITO_top + h_disk + h_TDBC + 0.15
    
    pml_layers = [mp.PML(thickness=dpml, direction=mp.Z)]
    
    sources = [mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ex,
        center=mp.Vector3(0, 0, z_source),
        size=mp.Vector3(sx, sy, 0)
    )]
    
    # TDBC layer only (reference)
    geometry_tdbc = [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, z_glass_top - z_bottom),
                 center=mp.Vector3(0, 0, (z_glass_top + z_bottom)/2), material=glass),
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, h_ITO),
                 center=mp.Vector3(0, 0, z_glass_top + h_ITO/2), material=ITO),
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, h_TDBC),
                 center=mp.Vector3(0, 0, z_ITO_top + h_TDBC/2), material=TDBC)
    ]
    
    # With disk
    geometry = geometry_tdbc.copy()
    geometry.append(mp.Cylinder(
        radius=D/2 + h_TDBC,
        height=h_disk + h_TDBC,
        center=mp.Vector3(0, 0, z_ITO_top + (h_disk + h_TDBC)/2),
        material=TDBC
    ))
    geometry.append(mp.Cylinder(
        radius=D/2,
        height=h_disk,
        center=mp.Vector3(0, 0, z_ITO_top + h_disk/2),
        material=Al_Palik
    ))
    
    # Reference (TDBC only)
    sim_ref = mp.Simulation(
        cell_size=cell_size, geometry=geometry_tdbc, boundary_layers=pml_layers,
        sources=sources, resolution=resolution, k_point=mp.Vector3(0, 0, 0)
    )
    trans_ref = sim_ref.add_flux(fcen, df, nfreq,
        mp.FluxRegion(center=mp.Vector3(0, 0, z_trans), size=mp.Vector3(sx, sy, 0)))
    sim_ref.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(0, 0, z_trans), 1e-3))
    flux_ref = np.array(mp.get_fluxes(trans_ref))
    freqs = np.array(mp.get_flux_freqs(trans_ref))
    
    # Full simulation
    sim = mp.Simulation(
        cell_size=cell_size, geometry=geometry, boundary_layers=pml_layers,
        sources=sources, resolution=resolution, k_point=mp.Vector3(0, 0, 0)
    )
    trans = sim.add_flux(fcen, df, nfreq,
        mp.FluxRegion(center=mp.Vector3(0, 0, z_trans), size=mp.Vector3(sx, sy, 0)))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(0, 0, z_trans), 1e-3))
    flux = np.array(mp.get_fluxes(trans))
    
    T_norm = np.where(flux_ref > 0, flux / flux_ref, 1.0)
    T_norm = gaussian_filter1d(T_norm, sigma=2)
    T_norm = np.clip(T_norm, 0.3, 1.2)
    
    wavelengths_nm = 1 / freqs * 1000
    return wavelengths_nm, T_norm

def emission_model(wavelengths, T_norm):
    """
    Model emission enhancement.
    
    Emission enhancement = TDBC_emission_spectrum × (1 + Purcell_factor)
    We approximate Purcell factor from (1 - T_norm)
    """
    # TDBC emission spectrum (Stokes shifted)
    lambda_em = 600  # nm
    gamma_em = 40    # nm
    emission_spectrum = 1 / (1 + ((wavelengths - lambda_em) / gamma_em)**2)
    
    # Purcell enhancement (correlates with absorption/polariton modes)
    one_minus_T = 1 - T_norm
    one_minus_T = np.clip(one_minus_T, 0, 0.8)
    
    # Emission enhancement: baseline (1.0) + modulation from polaritons
    enhancement = 1.0 + 0.8 * one_minus_T * emission_spectrum
    enhancement = gaussian_filter1d(enhancement, sigma=2)
    
    return enhancement

# Run simulations
print("\nSimulating for 2D map diameters...")
results = {}
all_diameters = np.union1d(diameters_map, diameters_specific)

for D in all_diameters:
    print(f"  D = {D} nm...", end=" ", flush=True)
    wl, T_norm = simulate_T_norm(D)
    emission = emission_model(wl, T_norm)
    results[D] = {'wavelengths': wl, 'T_norm': T_norm, 'emission': emission}
    print("done")

wavelengths = wl

# ============================================================
# PLOTTING - Exact paper format
# ============================================================

fig = plt.figure(figsize=(15, 8))

# (a) 2D emission map
ax1 = fig.add_subplot(2, 3, 1)

emission_2d = np.array([results[D]['emission'] for D in diameters_map]).T
D_mesh, wl_mesh = np.meshgrid(diameters_map, wavelengths)

im = ax1.pcolormesh(D_mesh, wl_mesh, emission_2d, shading='gouraud',
                     cmap='YlOrRd', vmin=1.0, vmax=1.8)

# Pump wavelength
ax1.axhline(530, color='yellow', linestyle='--', linewidth=2)

# Axis labels matching paper
ax1.set_xlabel('Diameter (nm)', fontsize=11)
ax1.set_ylabel('Wavelength (nm)', fontsize=11)
ax1.set_title('(a)', loc='left', fontsize=12, fontweight='bold')
ax1.set_xlim(75, 205)
ax1.set_ylim(650, 500)  # Inverted as in paper

# Custom ticks like paper
ax1.set_xticks([85, 105, 125, 145, 165, 195])

cbar = plt.colorbar(im, ax=ax1)
cbar.set_ticks([1.2, 1.4, 1.6, 1.8, 2.0])

# (b-f) Individual spectra
panel_labels = ['b', 'c', 'd', 'e', 'f']
for i, D in enumerate(diameters_specific):
    ax = fig.add_subplot(2, 3, i + 2)
    
    wl = results[D]['wavelengths']
    T_norm = results[D]['T_norm']
    emission = results[D]['emission']
    one_minus_T = 1 - T_norm
    
    # Left axis: Emission enhancement (black solid)
    ax.plot(wl, emission, 'k-', linewidth=2)
    ax.set_ylabel('Emission Enhancement', fontsize=10)
    ax.set_ylim(0.95, 2.0)
    ax.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8])
    
    # Right axis: 1 - T_norm (blue dashed)
    ax2 = ax.twinx()
    ax2.plot(wl, one_minus_T, 'b--', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('1 - T_norm', fontsize=10, color='blue')
    ax2.set_ylim(-0.1, 0.7)
    ax2.set_yticks([0, 0.2, 0.4, 0.6])
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax.set_xlabel('Wavelength (nm)', fontsize=10)
    ax.set_title(f'({panel_labels[i]})  D = {D} nm', loc='left', fontsize=11, fontweight='bold')
    ax.set_xlim(400, 700)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig5_exact.png', dpi=200, bbox_inches='tight', facecolor='white')
print("\nSaved: fig5_exact.png")

np.savez('fig5_exact_data.npz',
         wavelengths=wavelengths,
         diameters_map=diameters_map,
         diameters_specific=diameters_specific,
         results=results)
print("Saved: fig5_exact_data.npz")

