import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom

# Set readable font sizes
plt.rc('font', size=16)

# Define the range for N
N = np.linspace(1, 20, 100)

# Define the functions
y1 = N**2
y2 = N**2 + N**3
y3 = N**2 + binom(N, 2) + 2*N

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(N, y1, label='Standard layer', linewidth=2, color='blue', linestyle='-')
plt.plot(N, y2, label='Triadic layer', linewidth=2, color='red', linestyle=':')
plt.plot(N, y3, label='Spectral triadic layer', linewidth=2, color='orange', linestyle='--')

# Add labels and title
plt.xlabel('Size of the layers')
plt.ylabel('Number of parameters')
# Plot only integer ticks on x-axis
plt.xticks(np.arange(1, 21, 2))
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('scaling_comparison.png')
