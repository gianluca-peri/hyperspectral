import numpy as np
import torch
import sympy as sp
from list_of_functions_file import x, y
import os
import json

def generate_dataloaders(func, dim=1, **kwargs):
    num_samples = kwargs.get('num_samples', 1000)
    batch_size = kwargs.get('batch_size', 32)
    noise_std = kwargs.get('noise_std', 0.0)
    min_interval = kwargs.get('min_interval', -1.0)
    max_interval = kwargs.get('max_interval', 1.0)
    # Generate random input data in range [-1, 1]
    X = np.random.uniform(min_interval, max_interval, (num_samples, dim)).astype(np.float32)  # Shape (num_samples, dim)

    # Compute output data using the provided function
    f_lambdified = sp.lambdify((x, y), func, modules=['numpy']) if dim == 2 \
        else sp.lambdify((x), func, modules=['numpy'])

    y_values = f_lambdified(X[:, 0], X[:, 1]).astype(np.float32)  if dim == 2 \
        else f_lambdified(X[:, 0]).astype(np.float32) # Shape (num_samples,)

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

    # Create DataLoaders # In torch serve incapsulare i dati in queste strutture per gestirne allineamento, batch, shuffle...
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train, test


def train_and_evaluate(model, train, test, **kwargs):
    epochs = kwargs.get("epochs", 100)
    learning_rate = kwargs.get("learning_rate", 1e-3)
    model_name = kwargs.get("model_name", "model")
    save_dir = kwargs.get("save_dir", "./Results")
    device = kwargs.get("device", "cuda")

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
        for inputs, targets in train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_train_loss / len(train.dataset)
        history["train_loss"].append(epoch_train_loss)
        if (epoch + 1) % 10 == 0:
            print(f"[{model_name}] Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.4f}")
        # Save per epoch
        with open(os.path.join(save_dir, f"{model_name}_history.json"), "w") as f:
            json.dump(history, f, indent=4)

    # Test at the end
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_test_loss += loss.item() * inputs.size(0)

    final_test_loss = running_test_loss / len(test.dataset)
    history["test_loss"] = final_test_loss
    print(f"[{model_name}] Final Test Loss: {final_test_loss:.4f}")

    with open(os.path.join(save_dir, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}.pth"))
    return model