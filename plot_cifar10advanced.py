import os
import numpy as np
import matplotlib.pyplot as plt
from lib.plotting_style import apply, FIGSIZE, DPI
from lib.utils import load_all_histories_of_a_model

apply()

BASE_DIR = "cifar10advanced"

MODELS = {
    "standard_mlp_mixer": ("Standard MLP-Mixer", "o", "tab:blue"),
    "spectral_triadic_mixer": ("Spectral Triadic MLP-Mixer", "s", "tab:orange")
}

models_histories = {name: load_all_histories_of_a_model(BASE_DIR, name) for name in MODELS.keys()}


SUMMARY_PANELS = [
    ("train_loss",         "Training Loss",         "Loss"),
    ("val_loss",           "Validation Loss",       "Loss"),
    ("val_accuracy",       "Validation Accuracy",   "Accuracy (%)")
]

# Make summary plot (one row, three columns, global legend centered below)
fig, axes = plt.subplots(1, len(SUMMARY_PANELS), figsize=(FIGSIZE[0]*3, FIGSIZE[1]), dpi=DPI)
for i, (metric_key, metric_name, y_label) in enumerate(SUMMARY_PANELS):
    ax = axes[i]
    for model_key, (model_name, marker, color) in MODELS.items():
        histories = models_histories[model_key]
        mean_curve = np.mean([history[metric_key] for history in histories], axis=0)
        std_curve = np.std([history[metric_key] for history in histories], axis=0)
        ax.plot(mean_curve, marker=marker, color=color, label=model_name)
        ax.fill_between(range(len(mean_curve)), mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2)
    ax.set_title(metric_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.grid(True)

# Global legend centered below the subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(MODELS), frameon=False)
fig.tight_layout(rect=[0, 0.12, 1, 1])  # leave extra space at the bottom for the legend
plt.savefig(os.path.join(BASE_DIR, "summary_plot.png"))
