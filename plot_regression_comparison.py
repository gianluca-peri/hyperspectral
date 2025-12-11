import os
import json
import matplotlib.pyplot as plt
import glob

def get_latest_run_dir(base_dir="regression"):
    runs = glob.glob(os.path.join(base_dir, "run_*"))
    if not runs:
        return None
    # Sort by run number
    runs.sort(key=lambda x: int(os.path.basename(x).split("_")[-1]))
    return runs[-1]

def plot_results():
    base_dir = "regression"
    run_dir = get_latest_run_dir(base_dir)
    
    if not run_dir:
        print("No run directories found in 'regression/'")
        return

    print(f"Processing results from: {run_dir}")

    # Get list of functions (subdirectories in run_dir)
    # Filter out non-directories
    functions = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
    
    for func_name in functions:
        func_dir = os.path.join(run_dir, func_name)
        
        data_points = []

        # Iterate over hidden dimensions
        # Structure: hidden_{dim}
        if not os.path.exists(func_dir):
            continue
            
        hidden_dirs = [d for d in os.listdir(func_dir) if d.startswith("hidden_")]
        
        for h_dir in hidden_dirs:
            try:
                dim = int(h_dir.split("_")[1])
            except ValueError:
                continue
                
            h_path = os.path.join(func_dir, h_dir)
            
            # Read MLP loss
            mlp_path = os.path.join(h_path, "mlp", "history.json")
            triadic_path = os.path.join(h_path, "triadic_mlp", "history.json")
            
            if os.path.exists(mlp_path) and os.path.exists(triadic_path):
                try:
                    with open(mlp_path, 'r') as f:
                        mlp_data = json.load(f)
                        mlp_loss = mlp_data.get("test_loss", None)
                    
                    with open(triadic_path, 'r') as f:
                        triadic_data = json.load(f)
                        triadic_loss = triadic_data.get("test_loss", None)
                        
                    if mlp_loss is not None and triadic_loss is not None:
                        data_points.append((dim, mlp_loss, triadic_loss))
                except json.JSONDecodeError:
                    print(f"Error reading JSON in {h_path}")
                    continue

        if not data_points:
            print(f"No complete data found for {func_name}")
            continue
            
        # Sort by dimension
        data_points.sort(key=lambda x: x[0])
        
        dims = [x[0] for x in data_points]
        m_losses = [x[1] for x in data_points]
        t_losses = [x[2] for x in data_points]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(dims, m_losses, marker='o', label='MLP')
        plt.plot(dims, t_losses, marker='s', label='Triadic MLP')
        
        plt.title(f"Test Loss vs Hidden Neurons - {func_name}")
        plt.xlabel("Number of Hidden Neurons")
        plt.ylabel("Test Loss (MSE)")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        save_path = os.path.join(base_dir, f"{func_name}_comparison.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved comparison plot for {func_name} to {save_path}")

if __name__ == "__main__":
    plot_results()
