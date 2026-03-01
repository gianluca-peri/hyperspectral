import json
import matplotlib.pyplot as plt
import os
import numpy as np

# Set readable font sizes
plt.rc('font', size=16)


def load_history(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def get_all_runs(base_dir="cifar100advanced"):
    runs = []
    if not os.path.exists(base_dir):
        return runs
    
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and entry.startswith("run_"):
            runs.append(full_path)
    return runs


def aggregate_data(runs):
    # Map folder names to internal keys
    aggregated = {
        "standard_mlp_mixer": {"train_loss": [], "val_loss": [], "val_accuracy": []},
        "spectral_triadic_mixer": {"train_loss": [], "val_loss": [], "val_accuracy": []}
    }
    
    for run_dir in runs:
        std_path = os.path.join(run_dir, "standard_mlp_mixer", "history.json")
        spectral_path = os.path.join(run_dir, "spectral_triadic_mixer", "history.json")
        
        if os.path.exists(std_path):
            data = load_history(std_path)
            if "train_loss" in data:
                aggregated["standard_mlp_mixer"]["train_loss"].append(data["train_loss"])
            if "val_loss" in data:
                aggregated["standard_mlp_mixer"]["val_loss"].append(data["val_loss"])
            if "val_accuracy" in data:
                aggregated["standard_mlp_mixer"]["val_accuracy"].append(data["val_accuracy"])
            
        if os.path.exists(spectral_path):
            data = load_history(spectral_path)
            if "train_loss" in data:
                aggregated["spectral_triadic_mixer"]["train_loss"].append(data["train_loss"])
            if "val_loss" in data:
                aggregated["spectral_triadic_mixer"]["val_loss"].append(data["val_loss"])
            if "val_accuracy" in data:
                aggregated["spectral_triadic_mixer"]["val_accuracy"].append(data["val_accuracy"])
            
    return aggregated


def plot_metric(ax, model_data, metric_name, label, marker, color):
    if not model_data[metric_name]:
        return
        
    try:
        data_matrix = np.array(model_data[metric_name])
    except ValueError:
        print(f"Warning: Inconsistent epoch lengths for {label} {metric_name}. Truncating to minimum length.")
        min_len = min(len(x) for x in model_data[metric_name])
        data_matrix = np.array([x[:min_len] for x in model_data[metric_name]])
    
    if data_matrix.size == 0:
        return

    mean = np.mean(data_matrix, axis=0)
    std = np.std(data_matrix, axis=0)
    n = data_matrix.shape[0]
    sem = std / np.sqrt(n) if n > 0 else 0
    
    epochs = range(len(mean))
    
    ax.plot(epochs, mean, label=label, marker=marker, color=color)
    ax.fill_between(epochs, mean - sem, mean + sem, alpha=0.2, color=color)


def main():
    base_dir = "cifar100advanced"
    runs = get_all_runs(base_dir)
    print(f"Found {len(runs)} runs: {runs}")
    
    if not runs:
        print("No runs found.")
        return

    data = aggregate_data(runs)

    # Create plot with three panels: Train Loss, Val Loss, Val Accuracy
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Colors
    c_std = 'tab:blue'
    c_spec = 'tab:orange'

    # Plot Train Loss
    plot_metric(ax1, data["standard_mlp_mixer"], "train_loss", "Standard MLP-Mixer", "o", c_std)
    plot_metric(ax1, data["spectral_triadic_mixer"], "train_loss", "Spectral Triadic Mixer", "^", c_spec)
    
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot Validation Accuracy
    plot_metric(ax2, data["standard_mlp_mixer"], "val_accuracy", "Standard MLP-Mixer", "o", c_std)
    plot_metric(ax2, data["spectral_triadic_mixer"], "val_accuracy", "Spectral Triadic Mixer", "^", c_spec)

    # Draw horizontal lines at the maximum validation accuracy reached
    def max_val_accuracy(model_data):
        vals = model_data.get("val_accuracy", [])
        if not vals:
            return None
        # flatten list of runs
        flat = [v for run in vals for v in run]
        return max(flat) if flat else None

    max_std = max_val_accuracy(data["standard_mlp_mixer"])
    max_spec = max_val_accuracy(data["spectral_triadic_mixer"])

    if max_std is not None:
        ax2.axhline(y=max_std, color=c_std, linestyle='--',
                    label=f"Std max: {max_std:.2f}%")
    if max_spec is not None:
        ax2.axhline(y=max_spec, color=c_spec, linestyle='--',
                    label=f"Spec max: {max_spec:.2f}%")

    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    # Plot Validation Loss (third panel)
    plot_metric(ax3, data["standard_mlp_mixer"], "val_loss", "Standard MLP-Mixer", "o", c_std)
    plot_metric(ax3, data["spectral_triadic_mixer"], "val_loss", "Spectral Triadic Mixer", "^", c_spec)

    ax3.set_title('Validation Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)

    # Save plot
    output_path = os.path.join(base_dir, "graph_cifar100advanced.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
