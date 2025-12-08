import json
import matplotlib.pyplot as plt
import os
import numpy as np

# Set readable font sizes
plt.rc('font', size=20)

def load_history(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_all_runs(base_dir="mnist"):
    runs = []
    if not os.path.exists(base_dir):
        return runs
    
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and entry.startswith("run_"):
            runs.append(full_path)
    return runs

def aggregate_data(runs):
    # Structure: model_name -> metric -> list of lists (epochs) or list of scalars
    models = ["direct_space_triadic", "spectral_triadic"]
    aggregated = {m: {"train_loss": [], "val_loss": [], "val_accuracy": []} for m in models}
    
    for run_dir in runs:
        for model in models:
            history_path = os.path.join(run_dir, model, "history.json")
            if os.path.exists(history_path):
                data = load_history(history_path)
                aggregated[model]["train_loss"].append(data["train_loss"])
                aggregated[model]["val_loss"].append(data["val_loss"])
                aggregated[model]["val_accuracy"].append(data["val_accuracy"])
            
    return aggregated

def plot_metric_curve(ax, model_data, metric_name, label, marker, color):
    if not model_data[metric_name]:
        return
        
    # Convert to numpy array: shape (n_runs, n_epochs)
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
    
    epochs = range(1, len(mean) + 1)
    
    ax.plot(epochs, mean, label=label, marker=marker, color=color)
    ax.fill_between(epochs, mean - sem, mean + sem, alpha=0.2, color=color)

def main():
    base_dir = "mnist"
    runs = get_all_runs(base_dir)
    print(f"Found {len(runs)} runs: {runs}")
    
    if not runs:
        print("No runs found.")
        return

    data = aggregate_data(runs)

    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Colors
    c_direct = 'tab:blue'
    c_spectral = 'tab:orange'
    
    # Plot Train Loss
    plot_metric_curve(ax1, data["direct_space_triadic"], "train_loss", "Direct Space Triadic", "o", c_direct)
    plot_metric_curve(ax1, data["spectral_triadic"], "train_loss", "Spectral Triadic", "^", c_spectral)
    
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot Validation Loss
    plot_metric_curve(ax2, data["direct_space_triadic"], "val_loss", "Direct Space Triadic", "o", c_direct)
    plot_metric_curve(ax2, data["spectral_triadic"], "val_loss", "Spectral Triadic", "^", c_spectral)
    
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    # Plot Validation Accuracy
    plot_metric_curve(ax3, data["direct_space_triadic"], "val_accuracy", "Direct Space Triadic", "o", c_direct)
    plot_metric_curve(ax3, data["spectral_triadic"], "val_accuracy", "Spectral Triadic", "^", c_spectral)
    
    ax3.set_title('Validation Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True)

    # Save plot
    output_path = os.path.join(base_dir, "comparison_triadic_mnist.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
