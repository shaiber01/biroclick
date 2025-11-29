#!/usr/bin/env python3
"""
Figure 3c,d - EXACT paper format
FDTD Transmission spectra of bare and coated nanodisk arrays

Paper specifications:
- X-axis: Diameter (nm) 80-200
- Y-axis: Wavelength (nm) 400-800 (inverted - 400 at top, 800 at bottom)
- Colormap: 0-1 transmission (white=1, dark=0)
- (c) bare disks, (d) coated disks
"""

import meep as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from meep.materials import Al
from scipy.ndimage import gaussian_filter1d

print("=" * 60)
print("FIGURE 3c,d - Transmission Maps (Paper Format)")
print("=" * 60)

# Parameters
diameters_nm = np.array([80, 100, 120, 140, 160, 180, 200])
h_disk = 0.040
h_ITO = 0.030
h_TDBC = 0.020
gap = 0.180

resolution = 60
sz = 2.0
dpml = 0.4

wl_min, wl_max = 0.4, 0.8
freq_min, freq_max = 1/wl_max, 1/wl_min
fcen = (freq_min + freq_max) / 2
df = freq_max - freq_min
nfreq = 150

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

def simulate_transmission(D_nm, with_tdbc=False):
    """Simulate normalized transmission."""
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
    
    # Reference geometry (glass + ITO)
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
        # TDBC shell (conformal)
        geometry.append(mp.Cylinder(
            radius=D/2 + h_TDBC,
            height=h_disk + h_TDBC,
            center=mp.Vector3(0, 0, z_ITO_top + (h_disk + h_TDBC)/2),
            material=TDBC
        ))
    
    # Al disk
    geometry.append(mp.Cylinder(
        radius=D/2,
        height=h_disk,
        center=mp.Vector3(0, 0, z_ITO_top + h_disk/2),
        material=Al
    ))
    
    # Reference simulation
    sim_ref = mp.Simulation(
        cell_size=cell_size, geometry=geometry_ref, boundary_layers=pml_layers,
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
    
    T = np.where(flux_ref > 0, flux / flux_ref, 1)
    T = gaussian_filter1d(T, sigma=2)
    T = np.clip(T, 0, 1)
    
    wavelengths_nm = 1 / freqs * 1000
    return wavelengths_nm, T

# Run simulations
print("\nSimulating bare nanodisks...")
T_bare = {}
for D in diameters_nm:
    print(f"  D = {D} nm...", end=" ", flush=True)
    wl, T = simulate_transmission(D, with_tdbc=False)
    T_bare[D] = T
    print("done")

print("\nSimulating TDBC-coated nanodisks...")
T_coated = {}
for D in diameters_nm:
    print(f"  D = {D} nm...", end=" ", flush=True)
    wl, T = simulate_transmission(D, with_tdbc=True)
    T_coated[D] = T
    print("done")

wavelengths = wl

# ============================================================
# PLOTTING - Exact paper format
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Create 2D arrays for pcolormesh
T_bare_2d = np.array([T_bare[D] for D in diameters_nm]).T
T_coated_2d = np.array([T_coated[D] for D in diameters_nm]).T

D_mesh, wl_mesh = np.meshgrid(diameters_nm, wavelengths)

# Figure 3c - Bare nanodisks
im1 = ax1.pcolormesh(D_mesh, wl_mesh, T_bare_2d, shading='gouraud',
                      cmap='gray_r', vmin=0, vmax=1)
ax1.set_xlabel('Diameter (nm)', fontsize=12)
ax1.set_ylabel('Wavelength (nm)', fontsize=12)
ax1.set_title('(c)  FDTD', loc='left', fontsize=12, fontweight='bold')
ax1.set_xlim(80, 200)
ax1.set_ylim(800, 400)  # Inverted as in paper
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Figure 3d - Coated nanodisks
im2 = ax2.pcolormesh(D_mesh, wl_mesh, T_coated_2d, shading='gouraud',
                      cmap='gray_r', vmin=0, vmax=1)
ax2.set_xlabel('Diameter (nm)', fontsize=12)
ax2.set_ylabel('Wavelength (nm)', fontsize=12)
ax2.set_title('(d)  FDTD', loc='left', fontsize=12, fontweight='bold')
ax2.set_xlim(80, 200)
ax2.set_ylim(800, 400)  # Inverted as in paper

# Add exciton line at 590nm
ax2.axhline(590, color='yellow', linestyle='--', linewidth=1.5, alpha=0.8)

cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

plt.tight_layout()
plt.savefig('fig3cd_exact.png', dpi=200, bbox_inches='tight', facecolor='white')
print("\nSaved: fig3cd_exact.png")

# Save data
np.savez('fig3cd_exact_data.npz',
         wavelengths=wavelengths,
         diameters_nm=diameters_nm,
         T_bare=T_bare_2d,
         T_coated=T_coated_2d)
print("Saved: fig3cd_exact_data.npz")

