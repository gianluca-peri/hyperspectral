"""
plot_mnist.py – generate all MNIST plots and save each as a separate PNG.

One PNG per metric, all 4 models on every plot:
  mnist/train_loss.png
  mnist/val_loss.png
  mnist/val_accuracy.png
  mnist/val_avg_confidence.png
  mnist/confidence_histogram.png
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spectral.layers import SpectralTriadic, DirectSpaceTriadic

BASE_DIR = "mnist"

# ---------------------------------------------------------------------------
# Model styles  (name → label, marker, color)
# ---------------------------------------------------------------------------
MODELS = {
    "direct_linear":                   ("Direct Linear",                "o",  "tab:blue"),
    "direct_space_triadic":            ("Direct Space Triadic",         "s",  "tab:green"),
    "spectral_triadic":                ("Spectral Triadic",             "^",  "tab:orange"),
    "spectral_triadic_trained_eigvec": ("Spectral Triadic (Train Eigv)", "D",  "tab:red"),
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_history(path):
    with open(path, "r") as f:
        return json.load(f)


def get_all_runs(base_dir=BASE_DIR):
    runs = []
    if not os.path.exists(base_dir):
        return runs
    for entry in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, entry)
        if os.path.isdir(full) and entry.startswith("run_"):
            runs.append(full)
    return runs


def plot_mean_sem(ax, data_lists, label, marker, color):
    """Plot mean ± SEM from a list of per-run epoch-length lists."""
    if not data_lists:
        return
    try:
        mat = np.array(data_lists)
    except ValueError:
        min_len = min(len(x) for x in data_lists)
        mat = np.array([x[:min_len] for x in data_lists])
    if mat.size == 0:
        return
    mean = np.mean(mat, axis=0)
    sem = np.std(mat, axis=0) / np.sqrt(mat.shape[0])
    epochs = range(1, len(mean) + 1)
    ax.plot(epochs, mean, label=label, marker=marker, color=color,
            markevery=max(1, len(mean) // 20))
    ax.fill_between(epochs, mean - sem, mean + sem, alpha=0.2, color=color)


def save_fig(fig, filename):
    out = os.path.join(BASE_DIR, filename)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Aggregate history across all runs
# ---------------------------------------------------------------------------

def aggregate_all(runs):
    metrics = ["train_loss", "val_loss", "val_accuracy", "val_avg_confidence", "learning_rate"]
    agg = {name: {m: [] for m in metrics} for name in MODELS}
    for run in runs:
        for name in MODELS:
            p = os.path.join(run, name, "history.json")
            if not os.path.exists(p):
                continue
            d = load_history(p)
            for metric in metrics:
                if metric in d:
                    agg[name][metric].append(d[metric])
    return agg


# ---------------------------------------------------------------------------
# Epoch-curve plots  (one PNG per metric)
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "train_loss":         ("Training Loss",                 "Loss"),
    "val_loss":           ("Validation Loss",               "Loss"),
    "val_accuracy":       ("Validation Accuracy",           "Accuracy (%)"),
    "val_avg_confidence": ("Validation Average Confidence", "Average Confidence"),
    "learning_rate":      ("Learning Rate",                 "Learning Rate"),
}

def plot_epoch_curves(agg):
    plt.rc("font", size=16)
    for metric, (title, ylabel) in METRIC_LABELS.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        for name, (label, marker, color) in MODELS.items():
            plot_mean_sem(ax, agg[name][metric], label, marker, color)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if metric == "learning_rate":
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.4)
        save_fig(fig, f"{metric}.png")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class DirectLinearPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer = nn.Linear(784, 10)

    def forward(self, x):
        return self.layer(self.flatten(x))


class DirectSpaceTriadicPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer = DirectSpaceTriadic(784, 10)

    def forward(self, x):
        return self.layer(self.flatten(x))


class TriadicPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer = SpectralTriadic(784, 10)

    def forward(self, x):
        return self.layer(self.flatten(x))


class TrainedEigvecTriadicPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer = SpectralTriadic(784, 10, train_triadic_eigenvectors=True)

    def forward(self, x):
        return self.layer(self.flatten(x))


MODEL_CLASSES = {
    "direct_linear":                   DirectLinearPerceptron,
    "direct_space_triadic":            DirectSpaceTriadicPerceptron,
    "spectral_triadic":                TriadicPerceptron,
    "spectral_triadic_trained_eigvec": TrainedEigvecTriadicPerceptron,
}


def get_confidences(model, loader, device):
    model.eval()
    confs = []
    with torch.no_grad():
        for images, _ in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)
            confs.extend(torch.max(probs, dim=1).values.cpu().numpy())
    return np.array(confs)


# ---------------------------------------------------------------------------
# Confidence histogram (one PNG, all 4 models)
# ---------------------------------------------------------------------------

def plot_confidence_histogram(run_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_loader = DataLoader(
        datasets.MNIST(root="./data", train=False, download=True, transform=transform),
        batch_size=64, shuffle=False,
    )

    plt.rc("font", size=16)
    fig, ax = plt.subplots(figsize=(9, 6))

    any_plotted = False
    for name, (label, marker, color) in MODELS.items():
        pth = os.path.join(run_dir, name, "model_final.pth")
        if not os.path.exists(pth):
            print(f"  Skipping {name} – model_final.pth not found")
            continue
        model = MODEL_CLASSES[name]().to(device)
        model.load_state_dict(torch.load(pth, map_location=device))
        confs = get_confidences(model, test_loader, device)
        ax.hist(confs, bins=50, alpha=0.5, label=label, color=color, density=True)
        any_plotted = True

    if not any_plotted:
        print("Skipping confidence histogram – no model weights found.")
        plt.close(fig)
        return

    ax.set_xlabel("Confidence (Max Softmax Probability)")
    ax.set_ylabel("Density (Log Scale)")
    ax.set_title("Confidence Histogram on MNIST Test Set")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, "confidence_histogram.png")


# ---------------------------------------------------------------------------
# Calibration delta: accuracy − confidence  (both in [0, 1])
# Negative  → model is overconfident (confidence > accuracy)
# Positive  → model is underconfident
# ---------------------------------------------------------------------------

def plot_calibration_delta(agg):
    plt.rc("font", size=16)
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, (label, marker, color) in MODELS.items():
        acc_lists  = agg[name]["val_accuracy"]
        conf_lists = agg[name]["val_avg_confidence"]
        if not acc_lists or not conf_lists:
            continue

        # Align lengths across runs
        try:
            acc_mat  = np.array(acc_lists)
            conf_mat = np.array(conf_lists)
        except ValueError:
            min_len  = min(min(len(x) for x in acc_lists),
                           min(len(x) for x in conf_lists))
            acc_mat  = np.array([x[:min_len] for x in acc_lists])
            conf_mat = np.array([x[:min_len] for x in conf_lists])

        # val_accuracy is %, val_avg_confidence is [0,1] → normalise accuracy
        delta_mat = acc_mat / 100.0 - conf_mat          # shape (n_runs, n_epochs)

        mean = np.mean(delta_mat, axis=0)
        sem  = np.std(delta_mat, axis=0) / np.sqrt(delta_mat.shape[0])
        epochs = range(1, len(mean) + 1)
        ax.plot(epochs, mean, label=label, marker=marker, color=color,
                markevery=max(1, len(mean) // 20))
        ax.fill_between(epochs, mean - sem, mean + sem, alpha=0.2, color=color)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Calibration Delta (Accuracy − Confidence)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Δ = Accuracy − Confidence")
    ax.legend()
    ax.grid(True, alpha=0.4)
    save_fig(fig, "calibration_delta.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    runs = get_all_runs()
    print(f"Found {len(runs)} run(s): {runs}")

    if not runs:
        print("No runs found – nothing to plot.")
        return

    agg = aggregate_all(runs)
    plot_epoch_curves(agg)
    plot_calibration_delta(agg)
    plot_confidence_histogram(run_dir=runs[0])


if __name__ == "__main__":
    main()
