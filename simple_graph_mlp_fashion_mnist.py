import json
import matplotlib.pyplot as plt
import os
import numpy as np

# Set readable font sizes
plt.rc('font', size=16)

def load_history(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_all_runs(base_dir="mlp_fashion_mnist"):
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
    # Excluded "triadic_nonlinear_mlp"
    models = ["mlp", "nonlinear_mlp", "triadic_mlp"]
    # Excluded "val_loss" from metrics we care about plotting, though we can still load it if we want.
    # But to be simple, let's just load what we need.
    metrics = ["train_loss", "val_accuracy"]
    
    aggregated = {model: {metric: [] for metric in metrics} for model in models}
    
    for run_dir in runs:
        for model in models:
            history_path = os.path.join(run_dir, model, "history.json")
            if os.path.exists(history_path):
                try:
                    data = load_history(history_path)
                    for metric in metrics:
                        if metric in data:
                            aggregated[model][metric].append(data[metric])
                except Exception as e:
                    print(f"Error loading {history_path}: {e}")
            
    return aggregated

def plot_metric(ax, model_data, metric_name, label, marker, color):
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
    
    epochs = range(len(mean))
    
    ax.plot(epochs, mean, label=label, marker=marker, color=color)
    ax.fill_between(epochs, mean - sem, mean + sem, alpha=0.2, color=color)

def main():
    base_dir = "mlp_fashion_mnist"
    runs = get_all_runs(base_dir)
    print(f"Found {len(runs)} runs: {runs}")
    
    if not runs:
        print(f"No runs found in {base_dir}.")
        return

    data = aggregate_data(runs)

    # Create plot - 2 subplots instead of 3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Define styles - Excluded triadic_nonlinear_mlp
    styles = {
        "mlp": {"label": "MLP", "marker": "o", "color": "tab:blue"},
        "nonlinear_mlp": {"label": "Non-Linear MLP", "marker": "s", "color": "tab:green"},
        "triadic_mlp": {"label": "Triadic MLP", "marker": "^", "color": "tab:orange"}
    }

    # Plot Train Loss
    for model, style in styles.items():
        plot_metric(ax1, data[model], "train_loss", style["label"], style["marker"], style["color"])
    
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot Validation Accuracy (ax2 instead of ax3)
    for model, style in styles.items():
        plot_metric(ax2, data[model], "val_accuracy", style["label"], style["marker"], style["color"])

    # Plot horizontal axline for human accuracy performance
    ax2.axhline(y=83.5, color='r', linestyle='--', label='Human Performance')
    
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    # Save plot
    output_path = os.path.join(base_dir, "simple_graph_mlp_fashion_mnist.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
