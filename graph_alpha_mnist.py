import json
import matplotlib.pyplot as plt
import os
import numpy as np
import re

# Set readable font sizes
plt.rc('font', size=16)

def load_history(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_all_runs(base_dir="alpha_mnist"):
    runs = []
    if not os.path.exists(base_dir):
        return runs
    
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and entry.startswith("run_"):
            runs.append(full_path)
    return runs

def aggregate_data(runs):
    # Structure: alpha -> {"test_accuracy": [], "test_loss": []}
    aggregated = {}
    
    for run_dir in runs:
        # Iterate over subdirectories in the run folder
        for entry in os.listdir(run_dir):
            full_path = os.path.join(run_dir, entry)
            if os.path.isdir(full_path) and entry.startswith("triadic_perceptron_alpha_"):
                # Extract alpha value
                try:
                    alpha_str = entry.replace("triadic_perceptron_alpha_", "")
                    alpha = float(alpha_str)
                except ValueError:
                    continue
                
                history_path = os.path.join(full_path, "history.json")
                if os.path.exists(history_path):
                    data = load_history(history_path)
                    
                    if alpha not in aggregated:
                        aggregated[alpha] = {"test_accuracy": [], "test_loss": []}
                    
                    # Check if keys exist in history
                    if "test_accuracy" in data:
                        aggregated[alpha]["test_accuracy"].append(data["test_accuracy"])
                    if "test_loss" in data:
                        aggregated[alpha]["test_loss"].append(data["test_loss"])
            
    return aggregated

def plot_metric(ax, aggregated_data, metric_name, ylabel, title, color):
    alphas = sorted(aggregated_data.keys())
    if not alphas:
        return

    means = []
    sems = []
    
    for alpha in alphas:
        values = aggregated_data[alpha][metric_name]
        if not values:
            means.append(0)
            sems.append(0)
            continue
            
        data_array = np.array(values)
        mean = np.mean(data_array)
        std = np.std(data_array)
        n = len(data_array)
        sem = std / np.sqrt(n) if n > 0 else 0
        
        means.append(mean)
        sems.append(sem)
    
    means = np.array(means)
    sems = np.array(sems)
    
    ax.plot(alphas, means, marker='o', color=color, label='Mean')
    ax.fill_between(alphas, means - sems, means + sems, alpha=0.2, color=color, label='SEM')
    
    ax.set_title(title)
    ax.set_xlabel('Alpha')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    # ax.legend() # Optional, might clutter if it's just one line

def main():
    base_dir = "alpha_mnist"
    runs = get_all_runs(base_dir)
    print(f"Found {len(runs)} runs in {base_dir}")
    
    if not runs:
        print("No runs found.")
        return

    data = aggregate_data(runs)
    
    if not data:
        print("No data found in runs.")
        return

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Colors
    c_acc = 'tab:blue'
    c_loss = 'tab:orange'

    # Plot Test Accuracy
    plot_metric(ax1, data, "test_accuracy", "Test Accuracy (%)", "Test Accuracy vs Alpha", c_acc)

    # Plot Test Loss
    plot_metric(ax2, data, "test_loss", "Test Loss", "Test Loss vs Alpha", c_loss)

    # Save plot
    output_path = os.path.join(base_dir, "graph_alpha_mnist.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
