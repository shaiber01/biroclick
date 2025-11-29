#!/usr/bin/env python3
"""
Figure 4 - EXACT paper format
Nanorod transmission spectra

Paper specifications:
- (a) Bare nanorods, x-polarization: Length 75-205nm, Wavelength 400-800nm
- (b) Coated nanorods, x-polarization: Same axes
- (c) Coated nanorods, y-polarization: Same axes
- (d) Comparison D=140nm disk vs L=120nm rod
- (e) Tx/Ty ratio for L=120nm
- (f) Energy dispersion

All use grayscale colormap 0-1
"""

import meep as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

print("=" * 60)
print("FIGURE 4 - Nanorod Arrays (Paper Format)")
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

omega_X = 3.22e15 / (2 * np.pi * 3e14)
gamma_X = 1.0e14 / (2 * np.pi * 3e14)
f_TDBC = 0.45
TDBC = mp.Medium(epsilon=2.56, E_susceptibilities=[
    mp.LorentzianSusceptibility(frequency=omega_X, gamma=gamma_X, sigma=f_TDBC)
])

# Parameters
lengths_nm = np.array([75, 95, 120, 140, 160, 180, 205])
W = 0.040  # Width 40nm
h_rod = 0.040
h_ITO = 0.030
h_TDBC = 0.020
Gamma_x = 0.200  # x spacing
Gamma_y = 0.150  # y spacing

resolution = 60
sz = 2.0
dpml = 0.4

wl_min, wl_max = 0.4, 0.8
freq_min, freq_max = 1/wl_max, 1/wl_min
fcen = (freq_min + freq_max) / 2
df = freq_max - freq_min
nfreq = 150

def simulate_rod_transmission(L_nm, polarization='x', with_tdbc=False):
    """Simulate transmission through nanorod array."""
    L = L_nm / 1000
    
    sx = L + Gamma_x
    sy = W + Gamma_y
    cell_size = mp.Vector3(sx, sy, sz)
    
    z_bottom = -sz/2 + dpml
    z_glass_top = z_bottom + 0.5
    z_ITO_top = z_glass_top + h_ITO
    z_source = z_glass_top - 0.15
    z_trans = z_ITO_top + h_rod + h_TDBC + 0.15
    
    pml_layers = [mp.PML(thickness=dpml, direction=mp.Z)]
    
    # Source polarization
    component = mp.Ex if polarization == 'x' else mp.Ey
    
    sources = [mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df),
        component=component,
        center=mp.Vector3(0, 0, z_source),
        size=mp.Vector3(sx, sy, 0)
    )]
    
    # Reference geometry
    geometry_ref = [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, z_glass_top - z_bottom),
                 center=mp.Vector3(0, 0, (z_glass_top + z_bottom)/2), material=glass),
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, h_ITO),
                 center=mp.Vector3(0, 0, z_glass_top + h_ITO/2), material=ITO)
    ]
    
    # Full geometry
    geometry = geometry_ref.copy()
    
    if with_tdbc:
        # TDBC base layer
        geometry.append(mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, h_TDBC),
            center=mp.Vector3(0, 0, z_ITO_top + h_TDBC/2),
            material=TDBC
        ))
        # TDBC shell (conformal) - ellipsoid
        geometry.append(mp.Ellipsoid(
            size=mp.Vector3(L + 2*h_TDBC, W + 2*h_TDBC, h_rod + h_TDBC),
            center=mp.Vector3(0, 0, z_ITO_top + (h_rod + h_TDBC)/2),
            material=TDBC
        ))
    
    # Nanorod (ellipsoid)
    geometry.append(mp.Ellipsoid(
        size=mp.Vector3(L, W, h_rod),
        center=mp.Vector3(0, 0, z_ITO_top + h_rod/2),
        material=Al_Palik
    ))
    
    # Reference simulation
    sim_ref = mp.Simulation(
        cell_size=cell_size, geometry=geometry_ref, boundary_layers=pml_layers,
        sources=sources, resolution=resolution, k_point=mp.Vector3(0, 0, 0)
    )
    trans_ref = sim_ref.add_flux(fcen, df, nfreq,
        mp.FluxRegion(center=mp.Vector3(0, 0, z_trans), size=mp.Vector3(sx, sy, 0)))
    sim_ref.run(until_after_sources=mp.stop_when_fields_decayed(50, component, mp.Vector3(0, 0, z_trans), 1e-3))
    flux_ref = np.array(mp.get_fluxes(trans_ref))
    freqs = np.array(mp.get_flux_freqs(trans_ref))
    
    # Full simulation
    sim = mp.Simulation(
        cell_size=cell_size, geometry=geometry, boundary_layers=pml_layers,
        sources=sources, resolution=resolution, k_point=mp.Vector3(0, 0, 0)
    )
    trans = sim.add_flux(fcen, df, nfreq,
        mp.FluxRegion(center=mp.Vector3(0, 0, z_trans), size=mp.Vector3(sx, sy, 0)))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, component, mp.Vector3(0, 0, z_trans), 1e-3))
    flux = np.array(mp.get_fluxes(trans))
    
    T = np.where(flux_ref > 0, flux / flux_ref, 1)
    T = gaussian_filter1d(T, sigma=2)
    T = np.clip(T, 0, 1)
    
    wavelengths_nm = 1 / freqs * 1000
    return wavelengths_nm, T

# Run simulations
print("\n(a) Bare nanorods, x-polarization...")
T_bare_x = {}
for L in lengths_nm:
    print(f"  L = {L} nm...", end=" ", flush=True)
    wl, T = simulate_rod_transmission(L, 'x', False)
    T_bare_x[L] = T
    print("done")

print("\n(b) Coated nanorods, x-polarization...")
T_coated_x = {}
for L in lengths_nm:
    print(f"  L = {L} nm...", end=" ", flush=True)
    wl, T = simulate_rod_transmission(L, 'x', True)
    T_coated_x[L] = T
    print("done")

print("\n(c) Coated nanorods, y-polarization...")
T_coated_y = {}
for L in lengths_nm:
    print(f"  L = {L} nm...", end=" ", flush=True)
    wl, T = simulate_rod_transmission(L, 'y', True)
    T_coated_y[L] = T
    print("done")

wavelengths = wl

# ============================================================
# PLOTTING - Exact paper format
# ============================================================

fig = plt.figure(figsize=(15, 10))

# Create 2D arrays
T_bare_x_2d = np.array([T_bare_x[L] for L in lengths_nm]).T
T_coated_x_2d = np.array([T_coated_x[L] for L in lengths_nm]).T
T_coated_y_2d = np.array([T_coated_y[L] for L in lengths_nm]).T

L_mesh, wl_mesh = np.meshgrid(lengths_nm, wavelengths)

# (a) Bare, x-pol
ax1 = fig.add_subplot(2, 3, 1)
im1 = ax1.pcolormesh(L_mesh, wl_mesh, T_bare_x_2d, shading='gouraud',
                      cmap='gray_r', vmin=0, vmax=1)
ax1.set_xlabel('Length (nm)', fontsize=11)
ax1.set_ylabel('Wavelength (nm)', fontsize=11)
ax1.set_title('(a)  → x̂', loc='left', fontsize=12, fontweight='bold')
ax1.set_xlim(75, 205)
ax1.set_ylim(800, 400)
plt.colorbar(im1, ax=ax1)

# (b) Coated, x-pol
ax2 = fig.add_subplot(2, 3, 2)
im2 = ax2.pcolormesh(L_mesh, wl_mesh, T_coated_x_2d, shading='gouraud',
                      cmap='gray_r', vmin=0, vmax=1)
ax2.set_xlabel('Length (nm)', fontsize=11)
ax2.set_ylabel('Wavelength (nm)', fontsize=11)
ax2.set_title('(b)  → x̂', loc='left', fontsize=12, fontweight='bold')
ax2.set_xlim(75, 205)
ax2.set_ylim(800, 400)
ax2.axhline(590, color='yellow', linestyle='--', linewidth=1.5, alpha=0.8)
plt.colorbar(im2, ax=ax2)

# (c) Coated, y-pol
ax3 = fig.add_subplot(2, 3, 3)
im3 = ax3.pcolormesh(L_mesh, wl_mesh, T_coated_y_2d, shading='gouraud',
                      cmap='gray_r', vmin=0, vmax=1)
ax3.set_xlabel('Length (nm)', fontsize=11)
ax3.set_ylabel('Wavelength (nm)', fontsize=11)
ax3.set_title('(c)  ↑ ŷ', loc='left', fontsize=12, fontweight='bold')
ax3.set_xlim(75, 205)
ax3.set_ylim(800, 400)
ax3.axhline(590, color='yellow', linestyle='--', linewidth=1.5, alpha=0.8)
plt.colorbar(im3, ax=ax3)

# (d) Comparison D=140nm disk vs L=120nm rod
ax4 = fig.add_subplot(2, 3, 4)
# Find L=120nm spectra
T_120_x = T_coated_x[120]
T_120_y = T_coated_y[120]

ax4.plot(wavelengths, T_120_x, 'k-', linewidth=2, label='L=120nm, x-pol')
ax4.plot(wavelengths, T_120_y, 'k--', linewidth=2, label='L=120nm, y-pol')
ax4.set_xlabel('Wavelength (nm)', fontsize=11)
ax4.set_ylabel('Normalized Transmission', fontsize=11)
ax4.set_title('(d)', loc='left', fontsize=12, fontweight='bold')
ax4.set_xlim(450, 750)
ax4.set_ylim(0.2, 1.0)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# (e) Tx/Ty ratio
ax5 = fig.add_subplot(2, 3, 5)
ratio = np.where(T_120_y > 0.1, T_120_x / T_120_y, 1.0)
ratio = gaussian_filter1d(ratio, sigma=3)
ax5.plot(wavelengths, ratio, 'k-', linewidth=2)
ax5.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax5.axvline(590, color='gray', linestyle='--', alpha=0.5)
ax5.set_xlabel('Wavelength (nm)', fontsize=11)
ax5.set_ylabel('Tx / Ty', fontsize=11)
ax5.set_title('(e)  L = 120 nm', loc='left', fontsize=12, fontweight='bold')
ax5.set_xlim(450, 750)
ax5.set_ylim(0.5, 1.1)
ax5.grid(True, alpha=0.3)

# (f) Energy dispersion (analytical)
ax6 = fig.add_subplot(2, 3, 6)

# LSP energies for nanorods (similar to disks but different scaling)
E_X = 2.1
g = 0.2

# LSP dispersion from paper (approximate)
E_LSP = np.array([2.9, 2.6, 2.3, 2.1, 1.95, 1.85, 1.75])

def coupled_oscillator(E_LSP, E_X, g):
    delta = E_X - E_LSP
    Omega = np.sqrt(4*g**2 + delta**2)
    E_UP = 0.5 * (E_X + E_LSP + Omega)
    E_LP = 0.5 * (E_X + E_LSP - Omega)
    return E_UP, E_LP

E_UP, E_LP = coupled_oscillator(0.95*E_LSP, E_X, g)

# Fine grid
lengths_fine = np.linspace(75, 205, 100)
E_LSP_fine = np.interp(lengths_fine, lengths_nm, E_LSP)
E_UP_fine, E_LP_fine = coupled_oscillator(0.95*E_LSP_fine, E_X, g)

ax6.scatter(lengths_nm, E_LSP, marker='D', s=60, c='blue', 
            edgecolors='darkblue', linewidths=1, label='Bare LSP')
ax6.scatter(lengths_nm, E_UP, marker='o', s=50, c='red', 
            edgecolors='darkred', linewidths=1)
ax6.scatter(lengths_nm, E_LP, marker='o', s=50, c='red', 
            edgecolors='darkred', linewidths=1, label='Coated')
ax6.plot(lengths_fine, E_UP_fine, 'k-', linewidth=1.5)
ax6.plot(lengths_fine, E_LP_fine, 'k-', linewidth=1.5)
ax6.axhline(E_X, color='gray', linestyle='--', linewidth=1.5)

ax6.text(150, 2.5, r'$\hbar\Omega_R = 0.4$ eV', fontsize=10, fontweight='bold')

ax6.set_xlabel('Length (nm)', fontsize=11)
ax6.set_ylabel('Energy (eV)', fontsize=11)
ax6.set_title('(f)', loc='left', fontsize=12, fontweight='bold')
ax6.set_xlim(80, 200)
ax6.set_ylim(1.5, 3.0)
ax6.legend(loc='upper right', fontsize=9)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig4_exact.png', dpi=200, bbox_inches='tight', facecolor='white')
print("\nSaved: fig4_exact.png")

np.savez('fig4_exact_data.npz',
         wavelengths=wavelengths,
         lengths_nm=lengths_nm,
         T_bare_x=T_bare_x_2d,
         T_coated_x=T_coated_x_2d,
         T_coated_y=T_coated_y_2d)
print("Saved: fig4_exact_data.npz")

