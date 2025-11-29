#!/usr/bin/env python3
"""
CORRECTED Figure 2b,c: Electric field enhancement maps
=======================================================

FIX: Nanorod should be ELLIPSE, not rectangle!

Paper says: "nanorods, modeled as ellipses, with the major axis 65 nm 
long and the minor axis 25 nm long"

Using mp.Ellipsoid instead of mp.Block for the nanorod.

Author: ReproAgent
"""

import meep as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from meep.materials import Al

print("=" * 60)
print("CORRECTED Figure 2b,c: Field Enhancement")
print("FIX: Nanorod as ELLIPSE (not rectangle)")
print("=" * 60)

# Parameters
wavelength = 0.530  # 530 nm
frequency = 1 / wavelength

h_disk = 0.040   # 40 nm height
h_ITO = 0.030    # 30 nm ITO
z_monitor = 0.010  # 10 nm above ITO

resolution = 100  # 10 nm

# Materials
glass = mp.Medium(epsilon=1.51**2)

omega_p_ITO = 1.78e15 / (2 * np.pi * 3e14)
gamma_ITO = 1.5e14 / (2 * np.pi * 3e14)
ITO = mp.Medium(epsilon=3.9, E_susceptibilities=[
    mp.DrudeSusceptibility(frequency=omega_p_ITO, gamma=gamma_ITO, sigma=1.0)
])

def get_field_enhancement(geometry_type='disk', D_nm=140, L_nm=65, W_nm=25):
    """
    CORRECTED: Uses Ellipsoid for nanorod instead of Block.
    """
    
    if geometry_type == 'disk':
        D = D_nm / 1000
        period = D + 0.180
        sx = sy = period
        print(f"\nNanodisk: D={D_nm}nm (Cylinder)")
    else:
        L = L_nm / 1000  # Major axis (along x)
        W = W_nm / 1000  # Minor axis (along y)
        sx = 0.200
        sy = 0.150
        print(f"\nNanorod: L={L_nm}nm × W={W_nm}nm (ELLIPSOID)")
    
    sz = 1.5
    dpml = 0.3
    
    cell_size = mp.Vector3(sx, sy, sz)
    pml_layers = [mp.PML(thickness=dpml, direction=mp.Z)]
    
    z_bottom = -sz/2 + dpml
    z_glass_top = z_bottom + 0.4
    z_ITO_top = z_glass_top + h_ITO
    z_source = z_glass_top - 0.1
    z_field = z_ITO_top + z_monitor
    
    sources = [mp.Source(
        src=mp.ContinuousSource(frequency=frequency),
        component=mp.Ex,
        center=mp.Vector3(0, 0, z_source),
        size=mp.Vector3(sx, sy, 0)
    )]
    
    geometry = [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, z_glass_top - z_bottom),
                 center=mp.Vector3(0, 0, (z_glass_top + z_bottom)/2), material=glass),
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, h_ITO),
                 center=mp.Vector3(0, 0, z_glass_top + h_ITO/2), material=ITO)
    ]
    
    if geometry_type == 'disk':
        # Cylinder for disk
        geometry.append(mp.Cylinder(
            radius=D/2,
            height=h_disk,
            center=mp.Vector3(0, 0, z_ITO_top + h_disk/2),
            material=Al
        ))
    else:
        # ELLIPSOID for nanorod (CORRECTED!)
        # Ellipsoid size is the full extent in each direction
        # Major axis L along x, minor axis W along y, height h_disk along z
        geometry.append(mp.Ellipsoid(
            size=mp.Vector3(L, W, h_disk),
            center=mp.Vector3(0, 0, z_ITO_top + h_disk/2),
            material=Al
        ))
    
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        boundary_layers=pml_layers,
        sources=sources,
        resolution=resolution,
        k_point=mp.Vector3(0, 0, 0)
    )
    
    # Run to steady state
    sim.run(until=50)
    
    # Get field
    field_region = mp.Volume(
        center=mp.Vector3(0, 0, z_field),
        size=mp.Vector3(sx, sy, 0)
    )
    
    Ex = sim.get_array(component=mp.Ex, vol=field_region)
    Ey = sim.get_array(component=mp.Ey, vol=field_region)
    Ez = sim.get_array(component=mp.Ez, vol=field_region)
    
    E_mag = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
    
    x = np.linspace(-sx/2, sx/2, E_mag.shape[0])
    y = np.linspace(-sy/2, sy/2, E_mag.shape[1])
    
    E0 = 1.0
    enhancement = E_mag / E0
    
    return x * 1000, y * 1000, enhancement

# Run simulations
print("\n" + "=" * 60)
print("Running simulations...")
print("=" * 60)

x_disk, y_disk, E_disk = get_field_enhancement('disk', D_nm=140)
x_rod, y_rod, E_rod = get_field_enhancement('rod', L_nm=65, W_nm=25)

# Plotting
print("\nGenerating plots...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Figure 2b: Nanodisk
ax = axes[0]
X, Y = np.meshgrid(x_disk, y_disk)
im = ax.pcolormesh(X, Y, E_disk.T, shading='auto', cmap='hot', vmin=0, vmax=8)
ax.set_xlabel('x (nm)', fontsize=11)
ax.set_ylabel('y (nm)', fontsize=11)
ax.set_title(f'(b) Nanodisk D=140nm\n|E/E₀| at λ=530nm', fontsize=11)
ax.set_aspect('equal')

# Draw disk outline (circle)
circle = plt.Circle((0, 0), 70, fill=False, color='cyan', linewidth=2, linestyle='--')
ax.add_patch(circle)

# Polarization arrow
ax.annotate('', xy=(80, 0), xytext=(-80, 0),
            arrowprops=dict(arrowstyle='->', color='white', lw=2))
ax.text(0, -100, '→ x', color='white', fontsize=10, ha='center')

plt.colorbar(im, ax=ax, label='|E/E₀|')

# Figure 2c: Nanorod (ELLIPSE!)
ax = axes[1]
X, Y = np.meshgrid(x_rod, y_rod)
im = ax.pcolormesh(X, Y, E_rod.T, shading='auto', cmap='hot', vmin=0, vmax=8)
ax.set_xlabel('x (nm)', fontsize=11)
ax.set_ylabel('y (nm)', fontsize=11)
ax.set_title(f'(c) Nanorod L=65nm, W=25nm\n|E/E₀| at λ=530nm (ELLIPSE)', fontsize=11)
ax.set_aspect('equal')

# Draw ELLIPSE outline (not rectangle!)
from matplotlib.patches import Ellipse
ellipse = Ellipse((0, 0), 65, 25, fill=False, color='cyan', linewidth=2, linestyle='--')
ax.add_patch(ellipse)

# Polarization arrow
ax.annotate('', xy=(60, 0), xytext=(-60, 0),
            arrowprops=dict(arrowstyle='->', color='white', lw=2))

plt.colorbar(im, ax=ax, label='|E/E₀|')

plt.tight_layout()
plt.savefig('fig2bc_corrected.png', dpi=200, bbox_inches='tight')
print("\nSaved: fig2bc_corrected.png")

# Analysis
print("\n" + "=" * 60)
print("COMPARISON: Rectangle vs Ellipse")
print("=" * 60)

print(f"\nNanodisk (D=140nm):")
print(f"  Shape: Cylinder (correct)")
print(f"  Max |E/E₀|: {np.max(E_disk):.1f}")

print(f"\nNanorod (65x25nm):")
print(f"  Shape: ELLIPSOID (corrected from Block)")
print(f"  Max |E/E₀|: {np.max(E_rod):.1f}")

print("\nKey difference:")
print("  - Rectangle has sharp corners → artificial field hotspots")
print("  - Ellipse has smooth curvature → realistic field distribution")
print("  - Paper explicitly says 'modeled as ellipses'")

print("\n" + "=" * 60)
print("Figure 2b,c CORRECTED complete!")
print("=" * 60)

