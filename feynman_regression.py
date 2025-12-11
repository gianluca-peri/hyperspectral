import os
import csv
import json
import torch
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from spectral.layers import SpectralTriadic

# Define MLP and TriadicMLP classes
class MLP(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim=1):
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    
class TriadicMLP(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim=1):
        super(TriadicMLP, self).__init__()
        self.layer1 = SpectralTriadic(input_dim, hidden_dim)
        self.layer2 = SpectralTriadic(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def get_run_dir(base_dir="feynman_regression"):
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

def generate_dataloaders(func_str, variables, ranges, num_samples=1000, batch_size=32, noise_std=0.0):
    # Create symbols
    syms = sp.symbols(variables)
    
    # Parse formula
    # Ensure pi is treated as a constant if it appears in the formula but not in variables
    local_dict = {'pi': sp.pi}
    expr = sp.sympify(func_str, locals=local_dict)
    
    f_lambdified = sp.lambdify(syms, expr, modules=['numpy'])
    
    # Generate X
    X_list = []
    for (v_min, v_max) in ranges:
        # Generate random values for this variable
        col = np.random.uniform(v_min, v_max, num_samples)
        X_list.append(col)
        
    X = np.column_stack(X_list).astype(np.float32)
    
    # Compute y
    # f_lambdified takes arguments as separate arrays: f(x1, x2, ...)
    args = [X[:, i] for i in range(len(variables))]
    y_values = f_lambdified(*args)
    
    # Handle case where output is a scalar (constant function)
    if np.isscalar(y_values):
        y_values = np.full(num_samples, y_values)
    
    y_values = y_values.astype(np.float32)
    
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

    return train_loader, test_loader, len(variables)

def train_and_evaluate(model, model_name, save_dir, train_loader, test_loader, epochs=100, learning_rate=1e-3, device='cuda'):
    print(f"\nStarting training for {model_name} in {save_dir}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "test_loss": 0.0,
        "test_accuracy": 0.0
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
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_test_loss += loss.item() * inputs.size(0)
            all_targets.append(targets)
            all_outputs.append(outputs)

    final_test_loss = running_test_loss / len(test_loader.dataset)
    
    # Calculate R2 score as accuracy
    all_targets = torch.cat(all_targets)
    all_outputs = torch.cat(all_outputs)
    ss_res = torch.sum((all_targets - all_outputs) ** 2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
    r2_score = 1 - ss_res / (ss_tot + 1e-8)

    history["test_loss"] = final_test_loss
    history["test_accuracy"] = r2_score.item()
    print(f"[{model_name}] Final Test Loss: {final_test_loss:.4f}, Test Accuracy (R2): {r2_score.item():.4f}")

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pth"))
    print(f"Finished training {model_name}")

def parse_feynman_csv(filepath):
    equations = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) # Skip header
        
        # Header structure based on file read:
        # Filename,Number,Output,Formula,# variables,v1_name,v1_low,v1_high,v2_name,...
        
        for row in reader:
            if not row: continue
            
            filename = row[0]
            formula = row[3]
            try:
                num_vars = int(row[4])
            except ValueError:
                continue # Skip invalid rows
                
            variables = []
            ranges = []
            
            # Variables start at index 5
            # Each variable has 3 columns: name, low, high
            current_idx = 5
            for _ in range(num_vars):
                v_name = row[current_idx]
                try:
                    v_low = float(row[current_idx+1])
                    v_high = float(row[current_idx+2])
                except ValueError:
                    # Handle missing ranges if any, default to 1-5 maybe?
                    v_low = 1.0
                    v_high = 5.0
                
                variables.append(v_name)
                ranges.append((v_low, v_high))
                current_idx += 3
            
            equations.append({
                "name": filename,
                "formula": formula,
                "variables": variables,
                "ranges": ranges
            })
    return equations

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    run_dir = get_run_dir()
    print(f"Saving results to {run_dir}")
    
    equations = parse_feynman_csv("feynman_formulas.csv")
    
    hidden_dimensions = [10, 9, 8, 7, 6, 5, 4, 3, 2]
    
    for eq in equations:
        func_name = eq["name"]
        func_expr = eq["formula"]
        variables = eq["variables"]
        ranges = eq["ranges"]
        
        print(f"\nProcessing function: {func_name} ({func_expr})")
        
        try:
            train_loader, test_loader, input_dim = generate_dataloaders(func_expr, variables, ranges)
        except Exception as e:
            print(f"Error generating data for {func_name}: {e}")
            continue
        
        for h_dim in hidden_dimensions:
            print(f"  Hidden Dimension: {h_dim}")
            
            # Train MLP
            mlp_model = MLP(hidden_dim=h_dim, input_dim=input_dim).to(device)
            mlp_save_dir = os.path.join(run_dir, func_name, f"hidden_{h_dim}", "mlp")
            train_and_evaluate(mlp_model, f"MLP_h{h_dim}_{func_name}", mlp_save_dir, train_loader, test_loader, device=device)
            
            # Train TriadicMLP
            triadic_model = TriadicMLP(hidden_dim=h_dim, input_dim=input_dim).to(device)
            triadic_save_dir = os.path.join(run_dir, func_name, f"hidden_{h_dim}", "triadic_mlp")
            train_and_evaluate(triadic_model, f"TriadicMLP_h{h_dim}_{func_name}", triadic_save_dir, train_loader, test_loader, device=device)
