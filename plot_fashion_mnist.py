"""
plot_fashion_mnist.py – generate all Fashion-MNIST plots and save each as a separate PNG.

One PNG per metric, all 4 models on every plot:
  fashion_mnist/train_loss.png
  fashion_mnist/val_loss.png
  fashion_mnist/val_accuracy.png
  fashion_mnist/val_avg_confidence.png
  fashion_mnist/confidence_histogram.png
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.style import apply, FIGSIZE, DPI, fix_legend

apply()

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spectral.layers import SpectralTriadic, DirectSpaceTriadic

BASE_DIR = "fashion_mnist"

# ---------------------------------------------------------------------------
# Model styles  (name → label, marker, color)
# ---------------------------------------------------------------------------
MODELS = {
    "direct_linear":                   ("Direct Space Linear",                "o",  "tab:blue"),
    "direct_space_triadic":            ("Direct Space Triadic",          "s",  "tab:green"),
    "spectral_triadic":                ("Spectral Triadic",             "^",  "tab:orange"),
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
    fig.savefig(out, dpi=DPI)
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
    "val_avg_confidence": ("Validation Avg. Conf.", "Confidence"),
    "learning_rate":      ("Learning Rate",                 "Learning Rate"),
}

def plot_epoch_curves(agg):
    for metric, (title, ylabel) in METRIC_LABELS.items():
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for name, (label, marker, color) in MODELS.items():
            plot_mean_sem(ax, agg[name][metric], label, marker, color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if metric == "learning_rate":
            ax.set_yscale("log")
        _loc = 'upper right' if metric in ('train_loss', 'val_loss') else 'lower right'
        fix_legend(ax, loc=_loc)
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


def compute_ece(model, loader, device, n_bins=15):
    """Expected Calibration Error over n_bins equal-width confidence bins."""
    model.eval()
    confidences, corrects = [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)
            conf, preds = torch.max(probs, dim=1)
            confidences.extend(conf.cpu().numpy())
            corrects.extend((preds == labels.to(device)).cpu().numpy())
    confidences = np.array(confidences)
    corrects    = np.array(corrects, dtype=float)
    n           = len(confidences)
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(corrects[mask].mean() - confidences[mask].mean())
    return ece


# ---------------------------------------------------------------------------
# Confidence histogram (one PNG, all 4 models)
# ---------------------------------------------------------------------------

def plot_confidence_histogram(run_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    test_loader = DataLoader(
        datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform),
        batch_size=64, shuffle=False,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)

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
    ax.set_yscale("log")
    fix_legend(ax, loc='upper right')
    ax.grid(True, alpha=0.3)
    save_fig(fig, "test_confidence_histogram.png")


# ---------------------------------------------------------------------------
# ECE reliability diagram – binned accuracy vs confidence
# ---------------------------------------------------------------------------

def plot_ece(runs, n_bins=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    test_loader = DataLoader(
        datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform),
        batch_size=64, shuffle=False,
    )

    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # per_run_acc[name] = list of (n_bins,) arrays, one per run
    per_run_acc = {name: [] for name in MODELS}
    per_run_ece = {name: [] for name in MODELS}

    for run in runs:
        for name in MODELS:
            pth = os.path.join(run, name, "model_final.pth")
            if not os.path.exists(pth):
                continue
            model = MODEL_CLASSES[name]().to(device)
            model.load_state_dict(torch.load(pth, map_location=device))
            model.eval()

            confidences, corrects = [], []
            with torch.no_grad():
                for images, labels in test_loader:
                    probs = torch.softmax(model(images.to(device)), dim=1)
                    conf, preds = torch.max(probs, dim=1)
                    confidences.extend(conf.cpu().numpy())
                    corrects.extend((preds == labels.to(device)).cpu().numpy())
            confidences = np.array(confidences)
            corrects    = np.array(corrects, dtype=float)
            n           = len(confidences)

            bin_accs = []
            ece = 0.0
            for i in range(n_bins):
                mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
                if mask.sum() == 0:
                    bin_accs.append(np.nan)
                else:
                    acc_b  = corrects[mask].mean()
                    conf_b = confidences[mask].mean()
                    bin_accs.append(acc_b)
                    ece += (mask.sum() / n) * abs(acc_b - conf_b)
            per_run_acc[name].append(bin_accs)
            per_run_ece[name].append(ece)

    names_with_data = [name for name in MODELS if per_run_acc[name]]
    if not names_with_data:
        print("Skipping ECE plot – no model weights found.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Perfect‑calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

    for name in names_with_data:
        label, marker, color = MODELS[name]
        acc_mat  = np.array(per_run_acc[name])   # (n_runs, n_bins)
        ece_mean = np.mean(per_run_ece[name])
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_acc = np.nanmean(acc_mat, axis=0)
            sem_acc  = np.nanstd(acc_mat, axis=0) / np.sqrt(acc_mat.shape[0])
        ax.plot(bin_centers, mean_acc,
                label=f"{label} (ECE={ece_mean:.3f})",
                marker=marker, color=color, linewidth=1.5)
        ax.fill_between(bin_centers, mean_acc - sem_acc, mean_acc + sem_acc,
                        alpha=0.2, color=color)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    fix_legend(ax)
    ax.grid(True, alpha=0.4)
    save_fig(fig, "test_ece.png")


# ---------------------------------------------------------------------------
# Calibration delta: accuracy − confidence  (both in [0, 1])
# Negative  → model is overconfident (confidence > accuracy)
# Positive  → model is underconfident
# ---------------------------------------------------------------------------

def plot_calibration_delta(agg):
    fig, ax = plt.subplots(figsize=FIGSIZE)

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
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Δ = Accuracy − Confidence")
    fix_legend(ax)
    ax.grid(True, alpha=0.4)
    save_fig(fig, "val_calibration_delta.png")


# ---------------------------------------------------------------------------
# 4-panel summary figure (shared legend)
# ---------------------------------------------------------------------------

SUMMARY_PANELS = [
    ("train_loss",         "Training Loss",                 "Loss"),
    ("val_loss",           "Validation Loss",               "Loss"),
    ("val_accuracy",       "Validation Accuracy",           "Accuracy (%)"),
    ("val_avg_confidence", "Validation Avg. Conf.", "Confidence"),
]

def plot_summary_grid(agg):
    cell = FIGSIZE[0]   # each panel has the same width/height as a solo plot
    fig, axes = plt.subplots(1, 4, figsize=(cell * 4, cell))

    for ax, (metric, title, ylabel) in zip(axes, SUMMARY_PANELS):
        for name, (label, marker, color) in MODELS.items():
            plot_mean_sem(ax, agg[name][metric], label, marker, color)
        ax.set_title(title, pad=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.4)

    # Collect handles/labels from the first panel (all panels share models)
    handles, labels = axes[0].get_legend_handles_labels()

    # Place a single legend centred below all panels
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.12),
        frameon=True,
        fontsize=matplotlib.rcParams.get('axes.labelsize', 34),
    )

    out = os.path.join(BASE_DIR, "summary.png")
    #plt.suptitle('Fashion-MNIST')
    plt.tight_layout(w_pad=1.0)
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


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
    plot_summary_grid(agg)
    plot_calibration_delta(agg)
    plot_confidence_histogram(run_dir=runs[0])
    plot_ece(runs)


if __name__ == "__main__":
    main()
