import json
import matplotlib.pyplot as plt
import os
import numpy as np

# Set readable font sizes
plt.rc('font', size=16)

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
    # Structure: model_name -> metric -> list of lists (epochs)
    aggregated = {
        "direct_linear": {"train_loss": [], "val_accuracy": []},
        "spectral_triadic": {"train_loss": [], "val_accuracy": []}
    }
    
    for run_dir in runs:
        direct_path = os.path.join(run_dir, "direct_linear", "history.json")
        triadic_path = os.path.join(run_dir, "spectral_triadic", "history.json")
        
        if os.path.exists(direct_path):
            data = load_history(direct_path)
            aggregated["direct_linear"]["train_loss"].append(data["train_loss"])
            aggregated["direct_linear"]["val_accuracy"].append(data["val_accuracy"])
            
        if os.path.exists(triadic_path):
            data = load_history(triadic_path)
            aggregated["spectral_triadic"]["train_loss"].append(data["train_loss"])
            aggregated["spectral_triadic"]["val_accuracy"].append(data["val_accuracy"])
            
    return aggregated

def plot_metric(ax, model_data, metric_name, label, marker, color):
    if not model_data[metric_name]:
        return
        
    # Convert to numpy array: shape (n_runs, n_epochs)
    # Note: assumes all runs have same number of epochs.
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
    base_dir = "mnist"
    runs = get_all_runs(base_dir)
    print(f"Found {len(runs)} runs: {runs}")
    
    if not runs:
        print("No runs found.")
        return

    data = aggregate_data(runs)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Colors
    c_linear = 'tab:blue'
    c_triadic = 'tab:orange'

    # Plot Train Loss
    plot_metric(ax1, data["direct_linear"], "train_loss", "Direct Linear", "o", c_linear)
    plot_metric(ax1, data["spectral_triadic"], "train_loss", "Spectral Triadic", "^", c_triadic)
    
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot Validation Accuracy
    plot_metric(ax2, data["direct_linear"], "val_accuracy", "Direct Linear", "o", c_linear)
    plot_metric(ax2, data["spectral_triadic"], "val_accuracy", "Spectral Triadic", "^", c_triadic)
    
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    # Save plot
    output_path = os.path.join(base_dir, "simple_graph_mnist.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
