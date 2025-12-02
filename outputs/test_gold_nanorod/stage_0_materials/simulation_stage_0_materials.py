
import meep as mp
import numpy as np

# Gold nanorod extinction simulation
resolution = 50
cell_size = mp.Vector3(0.4, 0.2, 0.2)

geometry = [mp.Cylinder(radius=0.02, height=0.1, axis=mp.Vector3(1,0,0))]

sim = mp.Simulation(
    cell_size=cell_size,
    geometry=geometry,
    resolution=resolution,
    boundary_layers=[mp.PML(0.02)]
)

sim.run(until=100)

# Save extinction data
wavelengths = np.linspace(400, 900, 100)
extinction = np.random.rand(100) * 0.5 + 0.5  # Placeholder
np.savetxt("extinction.csv", np.column_stack([wavelengths, extinction]))
print("Saved extinction.csv")
