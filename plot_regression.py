import os
import json
import numpy as np
import matplotlib.pyplot as plt
from lib.plotting_style import apply, FIGSIZE, DPI

apply()

BASE_DIR = "regression"
MODEL_NAME = "spectral_triadic_mlp"
RUN_NUMBER = 0

run_dir = os.path.join(BASE_DIR, f"run_{RUN_NUMBER}")

reconstruction_file = os.path.join(run_dir, MODEL_NAME, "regression_reconstruction.json")

with open(reconstruction_file, 'r') as f:
    data = json.load(f)

inputs = np.array(data['inputs'])
true_profile = np.array(data['targets'])
reconstructions = np.array(data['reconstructions'])

x = inputs[:, 0]
y = inputs[:, 1]
z_true = true_profile[:, 0]
z_recon = reconstructions[:, 0]

# Use tricontour to plot directly from scattered data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGSIZE[0]*2, FIGSIZE[1]), dpi=DPI)

# Plot contour lines for the true profile using tricontour
ax1.tricontour(x, y, z_true, levels=70, colors='black', linewidths=0.5)
ax1.set_title("True Profile")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Plot contour lines for the reconstruction using tricontour
ax2.tricontour(x, y, z_recon, levels=70, colors='black', linewidths=0.5)
ax2.set_title("Reconstruction")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

fig.tight_layout()
plt.savefig(os.path.join(run_dir, "reconstruction_contour.png"))

print(f"Saved reconstruction contour to {os.path.join(run_dir, 'reconstruction_contour.png')}")
