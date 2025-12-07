import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom

# Set readable font sizes
plt.rc('font', size=16)

# Define the range for N
N = np.linspace(1, 20, 100)

# Define the functions
y_blue = N**2
y_orange = N**2 + binom(N, 2) + N
y_black = 2 * N**2

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(N, y_blue, label='Standard layer', linewidth=2, color='blue', linestyle='-')
plt.plot(N, y_orange, label='Spectral triadic layer', linewidth=2, color='orange', linestyle='--')
plt.plot(N, y_black, label='$Two Standard Layers$', linewidth=2, color='black', linestyle='-.')

# Add labels and title
plt.xlabel('Size of the layers')
plt.ylabel('Number of parameters')
# Plot only integer ticks on x-axis
plt.xticks(np.arange(1, 21, 2))
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('comparison_2n2.png')
