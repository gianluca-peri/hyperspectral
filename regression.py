import os
import json
import torch
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from spectral.layers import SpectralTriadic

# We are setting input dimension equal to 2 and output dimension equal to 1

x, y = sp.symbols('x y')

list_of_functions = [
    (2*x + 3*y + 1, "linear"),
    (x*y + y**2 + x**2, "quadratic"),
    (x*y**2, "cubic"),
    (sp.exp(x) + sp.exp(y), "exponential"),
    (sp.sin(x) + sp.cos(y), "trigonometric"),
    (sp.log(x + 2) + sp.log(y + 2), "logarithmic")
]

def get_run_dir(base_dir="regression"):
    """
    Create a new run directory to save results.
    Returns the path to the new run directory.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    i = 0
    while True:
        run_dir = os.path.join(base_dir, f"run_{i}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            return run_dir
        i += 1

def generate_dataloaders(func, num_samples=1000, batch_size=32, noise_std=0.0):
    # Generate random input data in range [-1, 1]
    X = np.random.uniform(-1, 1, (num_samples, 2)).astype(np.float32)  # Shape (num_samples, 2)
    
    # Compute output data using the provided function
    f_lambdified = sp.lambdify((x, y), func, modules=['numpy'])
    y_values = f_lambdified(X[:, 0], X[:, 1]).astype(np.float32)  # Shape (num_samples,)
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, size=y_values.shape).astype(np.float32)
    y_noisy = y_values + noise
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y_noisy).unsqueeze(1)  # Shape (num_samples, 1)
    
    # Create train and test splits
    split_idx = int(0.8 * num_samples)
    X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]

    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def plot_function_3d(func, func_name, save_dir):
    """
    Plots the 3D surface of the function and saves it.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create a grid of points
    x_vals = np.linspace(-1, 1, 100)
    y_vals = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Evaluate the function
    f_lambdified = sp.lambdify((x, y), func, modules=['numpy'])
    Z = f_lambdified(X, Y)
    
    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    
    ax.set_title(f"Function: {func_name}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    save_path = os.path.join(save_dir, f"{func_name}_3d_plot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved 3D plot for {func_name} to {save_path}")

class MLP(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim=2, output_dim=1):
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    
class TriadicMLP(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim=2, output_dim=1):
        super(TriadicMLP, self).__init__()
        self.layer1 = SpectralTriadic(input_dim, hidden_dim)
        self.layer2 = SpectralTriadic(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
def train_and_evaluate(model, model_name, save_dir, train_loader, test_loader, epochs=100, learning_rate=1e-3, device='cuda'):
    print(f"\nStarting training for {model_name} in {save_dir}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "test_loss": 0.0
    }

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        history["train_loss"].append(epoch_train_loss)

        if (epoch + 1) % 10 == 0:
            print(f"[{model_name}] Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}")
            
        # Save per epoch
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=4)

    # Test at the end
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_test_loss += loss.item() * inputs.size(0)

    final_test_loss = running_test_loss / len(test_loader.dataset)
    history["test_loss"] = final_test_loss
    print(f"[{model_name}] Final Test Loss: {final_test_loss:.4f}")

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pth"))
    print(f"Finished training {model_name}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    run_dir = get_run_dir()
    print(f"Saving results to {run_dir}")
    
    hidden_dimensions = [10, 9, 8, 7, 6, 5, 4, 3, 2]
    
    for func_expr, func_name in list_of_functions:
        print(f"\nProcessing function: {func_name}")
        
        # Plot the function
        func_save_dir = "regression"
        plot_function_3d(func_expr, func_name, func_save_dir)
        
        train_loader, test_loader = generate_dataloaders(func_expr)
        
        for h_dim in hidden_dimensions:
            print(f"  Hidden Dimension: {h_dim}")
            
            # Train MLP
            mlp_model = MLP(hidden_dim=h_dim).to(device)
            mlp_save_dir = os.path.join(run_dir, func_name, f"hidden_{h_dim}", "mlp")
            train_and_evaluate(mlp_model, f"MLP_h{h_dim}_{func_name}", mlp_save_dir, train_loader, test_loader, device=device)
            
            # Train TriadicMLP
            triadic_model = TriadicMLP(hidden_dim=h_dim).to(device)
            triadic_save_dir = os.path.join(run_dir, func_name, f"hidden_{h_dim}", "triadic_mlp")
            train_and_evaluate(triadic_model, f"TriadicMLP_h{h_dim}_{func_name}", triadic_save_dir, train_loader, test_loader, device=device)
    



