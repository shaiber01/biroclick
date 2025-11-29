#!/usr/bin/env python3
"""
Figure 2a - EXACT paper format
TDBC Absorbance and Fluorescence spectra
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paper specifications:
# - X-axis: Wavelength (nm) 500-800
# - Y-axis: Absorbance (a.u.) and Fluorescence (a.u.)
# - Blue dashed = Absorbance
# - Red solid = Fluorescence
# - Inset shows TDBC chemical formula (skip)

# TDBC parameters from paper
lambda_abs = 590  # nm - absorption peak (2.1 eV)
lambda_em = 594   # nm - emission peak (Stokes shift ~0.007 eV = ~4nm)
gamma_abs = 16    # nm - linewidth ~0.066 eV ≈ 16nm at 590nm
gamma_em = 14     # nm - slightly narrower emission

wavelengths = np.linspace(500, 800, 500)

# Lorentzian lineshapes
absorbance = 1 / (1 + ((wavelengths - lambda_abs) / (gamma_abs/2))**2)
fluorescence = 1 / (1 + ((wavelengths - lambda_em) / (gamma_em/2))**2)

# Normalize to match paper's appearance (~10% scale)
absorbance = absorbance * 10
fluorescence = fluorescence * 9

# Create figure matching paper style
fig, ax = plt.subplots(figsize=(5, 4))

ax.plot(wavelengths, absorbance, 'b--', linewidth=2, label='Absorbance')
ax.plot(wavelengths, fluorescence, 'r-', linewidth=2, label='Fluorescence')

ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_ylabel('Absorbance / Fluorescence (a.u.)', fontsize=12)
ax.set_xlim(500, 800)
ax.set_ylim(0, 12)

# Add scale markers like paper
ax.text(505, 10.5, '10°', fontsize=10)
ax.text(505, 5.5, '5°', fontsize=10)

ax.legend(loc='upper right', fontsize=10)
ax.set_title('(a)', loc='left', fontsize=12, fontweight='bold')

# Match paper's clean style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig2a_exact.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: fig2a_exact.png")

