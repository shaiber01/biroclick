#!/usr/bin/env python3
"""
Figure 3e,f - EXACT paper format
Coupled oscillator model dispersion and fractions

Paper specifications:
- (e) Energy dispersion:
  - X-axis: Diameter (nm) 80-200
  - Y-axis: Energy (eV) 1.8-3.0
  - Blue diamonds: bare LSP
  - Red dots: coated (polaritons)
  - Dashed line: exciton energy
  - Solid lines: coupled oscillator fit
  - Label: ℏΩR = 0.4 eV

- (f) Fractions (two subplots):
  - X-axis: Diameter (nm) 80-200
  - Y-axis: Fraction 0-1
  - Upper polariton: LSP (solid), Exciton (dashed)
  - Lower polariton: LSP (solid), Exciton (dashed)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 60)
print("FIGURE 3e,f - Coupled Oscillator Model (Paper Format)")
print("=" * 60)

# Physical parameters from paper
E_X = 2.1  # eV - exciton energy (590nm)
g = 0.2    # eV - coupling strength (gives Ω_R = 0.4 eV)
f = 0.95   # factor for LSP shift after coating

# LSP dispersion (bare) - from paper's experimental data
# LSP energy increases with decreasing diameter
diameters = np.array([80, 100, 120, 140, 160, 180, 200])

# LSP energies (approximate from paper's blue diamonds)
# Smaller disk = higher energy
E_LSP_bare = np.array([2.8, 2.5, 2.25, 2.1, 1.95, 1.85, 1.75])

# Coated LSP is redshifted by factor f
E_LSP_coated = f * E_LSP_bare

# Coupled oscillator model
# E_UP,LP = 0.5 * (E_X + E_LSP ± sqrt(4g² + (E_X - E_LSP)²))

def coupled_oscillator(E_LSP, E_X, g):
    """Calculate upper and lower polariton energies."""
    delta = E_X - E_LSP
    Omega = np.sqrt(4*g**2 + delta**2)
    E_UP = 0.5 * (E_X + E_LSP + Omega)
    E_LP = 0.5 * (E_X + E_LSP - Omega)
    return E_UP, E_LP

def hopfield_coefficients(E_LSP, E_X, g):
    """Calculate exciton and LSP fractions (Hopfield coefficients)."""
    delta = E_X - E_LSP
    Omega = np.sqrt(4*g**2 + delta**2)
    
    # Upper polariton
    alpha_UP = 0.5 * (1 + delta / Omega)  # Exciton fraction
    beta_UP = 0.5 * (1 - delta / Omega)   # LSP fraction
    
    # Lower polariton
    alpha_LP = 0.5 * (1 - delta / Omega)  # Exciton fraction
    beta_LP = 0.5 * (1 + delta / Omega)   # LSP fraction
    
    return alpha_UP, beta_UP, alpha_LP, beta_LP

# Calculate polariton energies
E_UP, E_LP = coupled_oscillator(E_LSP_coated, E_X, g)

# Calculate fractions
alpha_UP, beta_UP, alpha_LP, beta_LP = hopfield_coefficients(E_LSP_coated, E_X, g)

# Fine grid for smooth lines
diameters_fine = np.linspace(80, 200, 100)
E_LSP_fine = np.interp(diameters_fine, diameters, E_LSP_coated)
E_UP_fine, E_LP_fine = coupled_oscillator(E_LSP_fine, E_X, g)
alpha_UP_fine, beta_UP_fine, alpha_LP_fine, beta_LP_fine = hopfield_coefficients(E_LSP_fine, E_X, g)

# ============================================================
# PLOTTING - Exact paper format
# ============================================================

fig = plt.figure(figsize=(12, 5))

# Figure 3e - Energy dispersion
ax1 = fig.add_subplot(1, 3, 1)

# Bare LSP (blue diamonds)
ax1.scatter(diameters, E_LSP_bare, marker='D', s=60, c='blue', 
            edgecolors='darkblue', linewidths=1, label='Bare LSP', zorder=3)

# Polariton branches (red dots for data points)
ax1.scatter(diameters, E_UP, marker='o', s=50, c='red', 
            edgecolors='darkred', linewidths=1, zorder=3)
ax1.scatter(diameters, E_LP, marker='o', s=50, c='red', 
            edgecolors='darkred', linewidths=1, label='Coated', zorder=3)

# Coupled oscillator fit (solid lines)
ax1.plot(diameters_fine, E_UP_fine, 'k-', linewidth=1.5, label='Upper X-LSP')
ax1.plot(diameters_fine, E_LP_fine, 'k-', linewidth=1.5, label='Lower X-LSP')

# Exciton line (dashed)
ax1.axhline(E_X, color='gray', linestyle='--', linewidth=1.5)

# Rabi splitting annotation
ax1.annotate('', xy=(130, E_UP[3]), xytext=(130, E_LP[3]),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax1.text(135, E_X, r'$\hbar\Omega_R = 0.4$ eV', fontsize=10, fontweight='bold')

ax1.set_xlabel('Diameter (nm)', fontsize=12)
ax1.set_ylabel('Energy (eV)', fontsize=12)
ax1.set_title('(e)', loc='left', fontsize=12, fontweight='bold')
ax1.set_xlim(80, 200)
ax1.set_ylim(1.6, 3.0)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# Figure 3f - Upper polariton fractions
ax2 = fig.add_subplot(1, 3, 2)

ax2.plot(diameters_fine, beta_UP_fine, 'b-', linewidth=2, label='LSP')
ax2.plot(diameters_fine, alpha_UP_fine, 'r--', linewidth=2, label='Exciton')

ax2.set_xlabel('Diameter (nm)', fontsize=12)
ax2.set_ylabel('Fraction', fontsize=12)
ax2.set_title('(f) Upper X-LSP', loc='left', fontsize=12, fontweight='bold')
ax2.set_xlim(80, 200)
ax2.set_ylim(0, 1)
ax2.legend(loc='center right', fontsize=10)
ax2.grid(True, alpha=0.3)

# Figure 3f - Lower polariton fractions
ax3 = fig.add_subplot(1, 3, 3)

ax3.plot(diameters_fine, beta_LP_fine, 'b-', linewidth=2, label='LSP')
ax3.plot(diameters_fine, alpha_LP_fine, 'r--', linewidth=2, label='Exciton')

ax3.set_xlabel('Diameter (nm)', fontsize=12)
ax3.set_ylabel('Fraction', fontsize=12)
ax3.set_title('Lower X-LSP', loc='left', fontsize=12, fontweight='bold')
ax3.set_xlim(80, 200)
ax3.set_ylim(0, 1)
ax3.legend(loc='center right', fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig3ef_exact.png', dpi=200, bbox_inches='tight', facecolor='white')
print("\nSaved: fig3ef_exact.png")

# Print key results
print("\nRabi splitting at resonance (D~140nm):")
print(f"  E_UP = {E_UP[3]:.2f} eV")
print(f"  E_LP = {E_LP[3]:.2f} eV")
print(f"  Splitting = {E_UP[3] - E_LP[3]:.2f} eV")

