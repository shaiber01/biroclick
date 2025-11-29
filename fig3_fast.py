#!/usr/bin/env python3
"""
FAST version: Reproduce Figure 3c,d (fewer diameters, lower resolution)
========================================================================

Reduced from 9 to 5 diameters, resolution 60 instead of 80.
Expected time: ~10-15 minutes instead of 30-40.

Author: ReproAgent
"""

import meep as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from meep.materials import Al
from scipy.ndimage import gaussian_filter1d
import time

print("=" * 70)
print("FAST Figure 3c,d: Reduced diameters and resolution")
print("=" * 70)

# Fewer diameters for speed
diameters_nm = np.array([80, 110, 140, 170, 200])
print(f"\nDiameters: {diameters_nm} nm ({len(diameters_nm)} values)")

# Geometry
h_disk = 0.040
h_ITO = 0.030
h_TDBC = 0.020
gap = 0.180

# Lower resolution for speed
resolution = 60  # 16.7 nm (was 80)
sz = 2.0
dpml = 0.4

wl_min, wl_max = 0.4, 0.8
freq_min, freq_max = 1/wl_max, 1/wl_min
fcen = (freq_min + freq_max) / 2
df = freq_max - freq_min
nfreq = 150

print(f"Resolution: {resolution} pts/µm ({1000/resolution:.1f} nm)")

# Materials
glass = mp.Medium(epsilon=1.51**2)

omega_p_ITO = 1.78e15 / (2 * np.pi * 3e14)
gamma_ITO = 1.5e14 / (2 * np.pi * 3e14)
ITO = mp.Medium(epsilon=3.9, E_susceptibilities=[
    mp.DrudeSusceptibility(frequency=omega_p_ITO, gamma=gamma_ITO, sigma=1.0)
])

# TDBC - CORRECTED
omega_X = 3.22e15 / (2 * np.pi * 3e14)
gamma_X = 1.0e14 / (2 * np.pi * 3e14)
f_TDBC = 0.45
TDBC = mp.Medium(epsilon=2.56, E_susceptibilities=[
    mp.LorentzianSusceptibility(frequency=omega_X, gamma=gamma_X, sigma=f_TDBC)
])

def simulate_disk(D_nm, with_tdbc=False):
    D = D_nm / 1000
    period = D + gap
    sx = sy = period
    cell_size = mp.Vector3(sx, sy, sz)
    
    z_bottom = -sz/2 + dpml
    z_glass_top = z_bottom + 0.5
    z_ITO_top = z_glass_top + h_ITO
    z_disk_top = z_ITO_top + h_disk
    z_source = z_glass_top - 0.15
    z_trans = z_disk_top + (h_TDBC if with_tdbc else 0) + 0.15
    
    pml_layers = [mp.PML(thickness=dpml, direction=mp.Z)]
    sources = [mp.Source(src=mp.GaussianSource(fcen, fwidth=df), component=mp.Ex,
                         center=mp.Vector3(0, 0, z_source), size=mp.Vector3(sx, sy, 0))]
    
    geometry_ref = [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, z_glass_top - z_bottom),
                 center=mp.Vector3(0, 0, (z_glass_top + z_bottom)/2), material=glass),
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, h_ITO),
                 center=mp.Vector3(0, 0, z_glass_top + h_ITO/2), material=ITO)
    ]
    
    geometry = geometry_ref.copy()
    
    if with_tdbc:
        geometry.append(mp.Block(size=mp.Vector3(mp.inf, mp.inf, h_TDBC),
                                 center=mp.Vector3(0, 0, z_ITO_top + h_TDBC/2), material=TDBC))
        geometry.append(mp.Cylinder(radius=D/2 + h_TDBC, height=h_disk + h_TDBC,
                                    center=mp.Vector3(0, 0, z_ITO_top + (h_disk + h_TDBC)/2), material=TDBC))
    
    geometry.append(mp.Cylinder(radius=D/2, height=h_disk,
                                center=mp.Vector3(0, 0, z_ITO_top + h_disk/2), material=Al))
    
    # Reference
    sim_ref = mp.Simulation(cell_size=cell_size, geometry=geometry_ref, boundary_layers=pml_layers,
                            sources=sources, resolution=resolution, k_point=mp.Vector3(0, 0, 0))
    trans_ref = sim_ref.add_flux(fcen, df, nfreq,
                                  mp.FluxRegion(center=mp.Vector3(0, 0, z_trans), size=mp.Vector3(sx, sy, 0)))
    sim_ref.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(0, 0, z_trans), 1e-3))
    flux_ref = np.array(mp.get_fluxes(trans_ref))
    freqs = np.array(mp.get_flux_freqs(trans_ref))
    
    # Full
    sim = mp.Simulation(cell_size=cell_size, geometry=geometry, boundary_layers=pml_layers,
                        sources=sources, resolution=resolution, k_point=mp.Vector3(0, 0, 0))
    trans = sim.add_flux(fcen, df, nfreq,
                         mp.FluxRegion(center=mp.Vector3(0, 0, z_trans), size=mp.Vector3(sx, sy, 0)))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(0, 0, z_trans), 1e-3))
    flux = np.array(mp.get_fluxes(trans))
    
    T = np.where(flux_ref > 0, flux / flux_ref, 0)
    T = np.clip(T, 0, 1.5)
    T = gaussian_filter1d(T, sigma=2)
    T = np.clip(T, 0, 1)
    
    return 1 / freqs * 1000, T

# Run simulations
print("\n" + "=" * 70)
print("Part 1: Bare Nanodisks")
print("=" * 70)

bare_T = []
wavelengths = None
t_start = time.time()

for i, D in enumerate(diameters_nm):
    print(f"\n[{i+1}/{len(diameters_nm)}] D = {D} nm (bare)...")
    t0 = time.time()
    wl, T = simulate_disk(D, with_tdbc=False)
    if wavelengths is None:
        wavelengths = wl
    bare_T.append(T)
    print(f"  Done in {time.time()-t0:.1f}s")

bare_T = np.array(bare_T)
print(f"\nBare disks total: {time.time()-t_start:.1f}s")

print("\n" + "=" * 70)
print("Part 2: TDBC-Coated Nanodisks")
print("=" * 70)

coated_T = []
t_start = time.time()

for i, D in enumerate(diameters_nm):
    print(f"\n[{i+1}/{len(diameters_nm)}] D = {D} nm (TDBC)...")
    t0 = time.time()
    wl, T = simulate_disk(D, with_tdbc=True)
    coated_T.append(T)
    print(f"  Done in {time.time()-t0:.1f}s")

coated_T = np.array(coated_T)
print(f"\nCoated disks total: {time.time()-t_start:.1f}s")

# Plotting - paper format
print("\n" + "=" * 70)
print("Creating plots...")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

D_mesh, wl_mesh = np.meshgrid(diameters_nm, wavelengths)

# Figure 3c: Bare
ax = axes[0]
im = ax.pcolormesh(D_mesh, wl_mesh, bare_T.T, shading='gouraud', cmap='hot', vmin=0, vmax=1)
ax.set_xlabel('Diameter (nm)', fontsize=12)
ax.set_ylabel('Wavelength (nm)', fontsize=12)
ax.set_title('(c) FDTD: Bare Nanodisks', fontsize=12)
ax.set_xlim(80, 200)
ax.set_ylim(400, 800)
ax.invert_yaxis()
plt.colorbar(im, ax=ax, label='Transmission')

# Figure 3d: Coated
ax = axes[1]
im = ax.pcolormesh(D_mesh, wl_mesh, coated_T.T, shading='gouraud', cmap='hot', vmin=0, vmax=1)
ax.axhline(590, color='cyan', linestyle='--', linewidth=2, alpha=0.8, label='Exciton')
ax.set_xlabel('Diameter (nm)', fontsize=12)
ax.set_ylabel('Wavelength (nm)', fontsize=12)
ax.set_title('(d) FDTD: TDBC-Coated Nanodisks', fontsize=12)
ax.set_xlim(80, 200)
ax.set_ylim(400, 800)
ax.invert_yaxis()
ax.legend(loc='upper right')
plt.colorbar(im, ax=ax, label='Transmission')

plt.tight_layout()
plt.savefig('fig3cd_fast.png', dpi=200, bbox_inches='tight')
print("\nSaved: fig3cd_fast.png")

np.savez('fig3cd_fast_data.npz', diameters_nm=diameters_nm, wavelengths_nm=wavelengths,
         bare_transmission=bare_T, coated_transmission=coated_T)
print("Saved: fig3cd_fast_data.npz")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

from scipy.signal import find_peaks

for i, D in enumerate(diameters_nm):
    valid = (wavelengths > 450) & (wavelengths < 750)
    T_valid = coated_T[i][valid]
    wl_valid = wavelengths[valid]
    peaks, _ = find_peaks(-T_valid, prominence=0.02, distance=5)
    if len(peaks) > 0:
        dips = wl_valid[peaks]
        print(f"D={D}nm: dips at {', '.join([f'{d:.0f}' for d in sorted(dips)[:3]])} nm")

print("\n" + "=" * 70)
print("Figure 3c,d (fast version) complete!")
print("=" * 70)

