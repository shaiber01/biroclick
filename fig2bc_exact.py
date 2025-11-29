#!/usr/bin/env python3
"""
Figure 2b,c - EXACT paper format
Electric field enhancement |E/E₀| maps

Paper specifications:
- (b) Nanodisk D=140nm: x,y from -100 to 100 nm, colorbar 1-6
- (c) Nanorod (ellipse 65nm × 25nm): x from -100 to 100, y from -50 to 50, colorbar 2-8
- Field monitors 10nm above ITO
- λ = 530nm (LSP resonance)
- Arrows show polarization direction
"""

import meep as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

print("=" * 60)
print("FIGURE 2b,c - Electric Field Enhancement (Paper Format)")
print("=" * 60)

# Use Palik Al for better accuracy
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

# Geometry parameters (µm)
h_disk = 0.040
h_ITO = 0.030
resolution = 100  # High resolution for near-field

def simulate_field(structure_type):
    """Simulate field enhancement for disk or rod."""
    
    if structure_type == 'disk':
        # Nanodisk D=140nm
        D = 0.140
        sx = sy = 0.320  # Cell size to show field
        extent_x = (-100, 100)
        extent_y = (-100, 100)
    else:
        # Nanorod (ellipse) 65nm × 25nm
        L = 0.065  # Major axis
        W = 0.025  # Minor axis
        sx, sy = 0.200, 0.150
        extent_x = (-100, 100)
        extent_y = (-50, 50)
    
    sz = 0.8
    dpml = 0.2
    
    cell_size = mp.Vector3(sx, sy, sz)
    pml_layers = [mp.PML(thickness=dpml, direction=mp.Z)]
    
    z_bottom = -sz/2 + dpml
    z_glass_top = z_bottom + 0.2
    z_ITO_top = z_glass_top + h_ITO
    
    # Geometry
    geometry = [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, z_glass_top - z_bottom),
                 center=mp.Vector3(0, 0, (z_glass_top + z_bottom)/2), material=glass),
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, h_ITO),
                 center=mp.Vector3(0, 0, z_glass_top + h_ITO/2), material=ITO),
    ]
    
    if structure_type == 'disk':
        geometry.append(mp.Cylinder(
            radius=D/2, height=h_disk,
            center=mp.Vector3(0, 0, z_ITO_top + h_disk/2),
            material=Al_Palik
        ))
    else:
        geometry.append(mp.Ellipsoid(
            size=mp.Vector3(L, W, h_disk),
            center=mp.Vector3(0, 0, z_ITO_top + h_disk/2),
            material=Al_Palik
        ))
    
    # CW source at 530nm (paper's LSP resonance wavelength)
    freq = 1 / 0.530
    sources = [mp.Source(
        src=mp.ContinuousSource(frequency=freq),
        component=mp.Ex,
        center=mp.Vector3(0, 0, z_glass_top - 0.05),
        size=mp.Vector3(sx, sy, 0)
    )]
    
    sim = mp.Simulation(
        cell_size=cell_size, geometry=geometry, boundary_layers=pml_layers,
        sources=sources, resolution=resolution, k_point=mp.Vector3(0, 0, 0)
    )
    
    # Monitor 10nm above ITO (as paper specifies)
    z_monitor = z_ITO_top + 0.010
    
    # Run to steady state
    sim.run(until=100)
    
    # Get field components
    Ex = sim.get_array(center=mp.Vector3(0, 0, z_monitor), 
                       size=mp.Vector3(sx, sy, 0), component=mp.Ex)
    Ey = sim.get_array(center=mp.Vector3(0, 0, z_monitor), 
                       size=mp.Vector3(sx, sy, 0), component=mp.Ey)
    Ez = sim.get_array(center=mp.Vector3(0, 0, z_monitor), 
                       size=mp.Vector3(sx, sy, 0), component=mp.Ez)
    
    # |E| magnitude
    E_mag = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
    
    # Get reference E0 (run without structure)
    sim_ref = mp.Simulation(
        cell_size=cell_size, 
        geometry=geometry[:2],  # Glass + ITO only
        boundary_layers=pml_layers,
        sources=sources, resolution=resolution, k_point=mp.Vector3(0, 0, 0)
    )
    sim_ref.run(until=100)
    
    Ex0 = sim_ref.get_array(center=mp.Vector3(0, 0, z_monitor), 
                            size=mp.Vector3(sx, sy, 0), component=mp.Ex)
    Ey0 = sim_ref.get_array(center=mp.Vector3(0, 0, z_monitor), 
                            size=mp.Vector3(sx, sy, 0), component=mp.Ey)
    Ez0 = sim_ref.get_array(center=mp.Vector3(0, 0, z_monitor), 
                            size=mp.Vector3(sx, sy, 0), component=mp.Ez)
    
    E0_mag = np.sqrt(np.abs(Ex0)**2 + np.abs(Ey0)**2 + np.abs(Ez0)**2)
    E0_mean = np.mean(E0_mag[E0_mag > 0])
    
    # |E/E₀|
    enhancement = E_mag / E0_mean
    enhancement = gaussian_filter(enhancement, sigma=1)
    
    return enhancement, extent_x, extent_y

# Run simulations
print("\nSimulating nanodisk (D=140nm)...")
enh_disk, ext_x_disk, ext_y_disk = simulate_field('disk')
print(f"  Max |E/E₀| = {np.max(enh_disk):.1f}")

print("\nSimulating nanorod (65nm × 25nm)...")
enh_rod, ext_x_rod, ext_y_rod = simulate_field('rod')
print(f"  Max |E/E₀| = {np.max(enh_rod):.1f}")

# ============================================================
# PLOTTING - Exact paper format
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Figure 2b - Nanodisk
im1 = ax1.imshow(enh_disk.T, origin='lower', cmap='hot',
                  extent=[ext_x_disk[0], ext_x_disk[1], ext_y_disk[0], ext_y_disk[1]],
                  vmin=1, vmax=6, aspect='equal')
ax1.set_xlabel('x (nm)', fontsize=12)
ax1.set_ylabel('y (nm)', fontsize=12)
ax1.set_title('(b)', loc='left', fontsize=12, fontweight='bold')
ax1.set_xlim(-100, 100)
ax1.set_ylim(-100, 100)

# Add polarization arrow
ax1.annotate('', xy=(80, 80), xytext=(50, 80),
             arrowprops=dict(arrowstyle='->', color='white', lw=2))

cbar1 = plt.colorbar(im1, ax=ax1, label='|E/E₀|', shrink=0.8)
cbar1.set_ticks([1, 2, 3, 4, 5, 6])

# Figure 2c - Nanorod
im2 = ax2.imshow(enh_rod.T, origin='lower', cmap='hot',
                  extent=[ext_x_rod[0], ext_x_rod[1], ext_y_rod[0], ext_y_rod[1]],
                  vmin=2, vmax=8, aspect='equal')
ax2.set_xlabel('x (nm)', fontsize=12)
ax2.set_ylabel('y (nm)', fontsize=12)
ax2.set_title('(c)', loc='left', fontsize=12, fontweight='bold')
ax2.set_xlim(-100, 100)
ax2.set_ylim(-50, 50)

# Add polarization arrow
ax2.annotate('', xy=(80, 35), xytext=(50, 35),
             arrowprops=dict(arrowstyle='->', color='white', lw=2))

cbar2 = plt.colorbar(im2, ax=ax2, label='|E/E₀|', shrink=0.8)
cbar2.set_ticks([2, 3, 4, 5, 6, 7, 8])

plt.tight_layout()
plt.savefig('fig2bc_exact.png', dpi=200, bbox_inches='tight', facecolor='white')
print("\nSaved: fig2bc_exact.png")

print(f"\nDisk max enhancement: {np.max(enh_disk):.1f} (paper: ~6)")
print(f"Rod max enhancement: {np.max(enh_rod):.1f} (paper: ~8)")

