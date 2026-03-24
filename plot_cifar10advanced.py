"""
plot_cifar10advanced.py – generate all CIFAR-10 Advanced plots and save each as a separate PNG.

One PNG per metric, both models on every plot:
  cifar10advanced/train_loss.png
  cifar10advanced/val_loss.png
  cifar10advanced/val_accuracy.png
  cifar10advanced/val_avg_confidence.png
  cifar10advanced/val_calibration_delta.png
  cifar10advanced/test_confidence_histogram.png
  cifar10advanced/test_ece.png
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
from spectral.layers import SpectralTriadic

BASE_DIR = "cifar10advanced"

# ---------------------------------------------------------------------------
# Model styles  (name → label, marker, color)
# ---------------------------------------------------------------------------
MODELS = {
    "standard_mlp_mixer":    ("Standard MLP-Mixer",    "o", "tab:blue"),
    "spectral_triadic_mixer": ("Spectral Triadic Mixer", "^", "tab:orange"),
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
        any_data = False
        for name, (label, marker, color) in MODELS.items():
            if agg[name][metric]:
                plot_mean_sem(ax, agg[name][metric], label, marker, color)
                any_data = True
        if not any_data:
            plt.close(fig)
            continue
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if metric == "learning_rate":
            ax.set_yscale("log")
        _loc = 'upper right' if metric in ('train_loss', 'val_loss') else 'lower right'
        fix_legend(ax, loc=_loc)
        ax.grid(True, alpha=0.4)
        save_fig(fig, f"{metric}.png")


# ---------------------------------------------------------------------------
# Model definitions (matching cifar10advanced.py)
# ---------------------------------------------------------------------------

class MixerBlock(nn.Module):
    def __init__(self, num_patches, hidden_dim,
                 tokens_mlp_dim=None, channels_mlp_dim=None):
        super().__init__()
        if tokens_mlp_dim is None:
            tokens_mlp_dim = num_patches * 4
        if channels_mlp_dim is None:
            channels_mlp_dim = hidden_dim * 4

        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_patches, tokens_mlp_dim),
            nn.GELU(),
            nn.Linear(tokens_mlp_dim, num_patches),
        )
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, channels_mlp_dim),
            nn.GELU(),
            nn.Linear(channels_mlp_dim, hidden_dim),
        )

    def forward(self, x):
        residual = x
        y = self.token_norm(x).transpose(1, 2)
        y = self.token_mlp(y)
        x = residual + y.transpose(1, 2)
        residual = x
        y = self.channel_norm(x)
        y = self.channel_mlp(y)
        x = residual + y
        return x


class StandardMLPMixer(nn.Module):
    def __init__(self, in_channels=3, img_size=32, patch_size=4,
                 hidden_dim=128, num_layers=6, num_classes=10,
                 tokens_mlp_dim=None, channels_mlp_dim=None):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.hidden_dim = hidden_dim
        self.patch_embed = nn.Linear(self.patch_dim, hidden_dim)
        self.blocks = nn.Sequential(*[
            MixerBlock(self.num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        x = x.unfold(2, P, P).unfold(3, P, P)
        x = x.contiguous().permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(B, self.num_patches, self.patch_dim)
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


class SpectralMixerBlock(nn.Module):
    def __init__(self, num_patches, hidden_dim):
        super().__init__()
        self.token_norm = nn.LayerNorm(num_patches)
        self.token_triadic = SpectralTriadic(num_patches, num_patches, bias=True)
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_triadic = SpectralTriadic(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        B, P, C = x.shape
        residual = x
        y = x.transpose(1, 2)
        y = self.token_norm(y)
        y = y.reshape(B * C, P)
        y = self.token_triadic(y)
        y = y.reshape(B, C, P).transpose(1, 2)
        x = residual + y
        residual = x
        y = self.channel_norm(x)
        y = y.reshape(B * P, C)
        y = self.channel_triadic(y)
        y = y.reshape(B, P, C)
        x = residual + y
        return x


class SpectralMLPMixer(nn.Module):
    def __init__(self, in_channels=3, img_size=32, patch_size=4,
                 hidden_dim=128, num_layers=6, num_classes=10):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.hidden_dim = hidden_dim
        self.patch_embed = nn.Linear(self.patch_dim, hidden_dim)
        self.blocks = nn.Sequential(*[
            SpectralMixerBlock(self.num_patches, hidden_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        x = x.unfold(2, P, P).unfold(3, P, P)
        x = x.contiguous().permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(B, self.num_patches, self.patch_dim)
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


MODEL_CLASSES = {
    "standard_mlp_mixer":    StandardMLPMixer,
    "spectral_triadic_mixer": SpectralMLPMixer,
}

# CIFAR-10 normalisation stats
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return DataLoader(
        datasets.CIFAR10(root="./data", train=False, download=True, transform=transform),
        batch_size=64, shuffle=False,
    )


def get_confidences(model, loader, device):
    model.eval()
    confs = []
    with torch.no_grad():
        for images, _ in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)
            confs.extend(torch.max(probs, dim=1).values.cpu().numpy())
    return np.array(confs)


# ---------------------------------------------------------------------------
# Confidence histogram (one PNG, all models)
# ---------------------------------------------------------------------------

def plot_confidence_histogram(run_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = get_test_loader()

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
    test_loader = get_test_loader()

    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

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
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

    for name in names_with_data:
        label, marker, color = MODELS[name]
        acc_mat  = np.array(per_run_acc[name])
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
# ---------------------------------------------------------------------------

def plot_calibration_delta(agg):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for name, (label, marker, color) in MODELS.items():
        acc_lists  = agg[name]["val_accuracy"]
        conf_lists = agg[name]["val_avg_confidence"]
        if not acc_lists or not conf_lists:
            continue

        try:
            acc_mat  = np.array(acc_lists)
            conf_mat = np.array(conf_lists)
        except ValueError:
            min_len  = min(min(len(x) for x in acc_lists),
                           min(len(x) for x in conf_lists))
            acc_mat  = np.array([x[:min_len] for x in acc_lists])
            conf_mat = np.array([x[:min_len] for x in conf_lists])

        # val_accuracy is %, val_avg_confidence is [0,1] → normalise accuracy
        delta_mat = acc_mat / 100.0 - conf_mat

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
# 3-panel summary figure (shared legend)
# ---------------------------------------------------------------------------

SUMMARY_PANELS = [
    ("train_loss",   "Training Loss",      "Loss"),
    ("val_loss",     "Validation Loss",    "Loss"),
    ("val_accuracy", "Validation Accuracy", "Accuracy (%)"),
]

def plot_summary_grid(agg):
    cell = FIGSIZE[0]
    fig, axes = plt.subplots(1, 3, figsize=(cell * 3, cell))

    for ax, (metric, title, ylabel) in zip(axes, SUMMARY_PANELS):
        for name, (label, marker, color) in MODELS.items():
            plot_mean_sem(ax, agg[name][metric], label, marker, color)
        ax.set_title(title, pad=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.12),
        frameon=True,
        fontsize=matplotlib.rcParams.get('axes.labelsize', 34),
    )

    out = os.path.join(BASE_DIR, "summary.png")
    plt.tight_layout(w_pad=2.0)
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
