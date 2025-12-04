import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from spectral.layers import SpectralLinear, SpectralTriadic

def get_run_dir(base_dir="mnist"):
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

class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # MNIST images are 28x28 = 784
        self.layer = SpectralLinear(784, 10)

    def forward(self, x):
        x = self.flatten(x)
        return self.layer(x)

class TriadicPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # MNIST images are 28x28 = 784
        self.layer = SpectralTriadic(784, 10)

    def forward(self, x):
        x = self.flatten(x)
        return self.layer(x)

# Hyperparameters
batch_size = 64
learning_rate = 1e-3
epochs = 20

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split train into train and validation (e.g., 50000 train, 10000 val)
train_size = 50000
val_size = 10000
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def train_and_evaluate(model, model_name, run_dir, train_loader, val_loader, test_loader, device, epochs, learning_rate):
    print(f"\nStarting training for {model_name}")
    model_dir = os.path.join(run_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "test_accuracy": 0.0,
        "test_loss": 0.0
    }
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        print(f"[{model_name}] Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_acc)
        
        # Save per epoch
        with open(os.path.join(model_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=4)

    # Test
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total

    print(f"[{model_name}] Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    history["test_loss"] = avg_test_loss
    history["test_accuracy"] = test_acc

    with open(os.path.join(model_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)
        
    torch.save(model.state_dict(), os.path.join(model_dir, "model_final.pth"))

# Setup save directory
run_dir = get_run_dir()
print(f"Saving results to {run_dir}")

# Train Linear Perceptron
linear_model = Perceptron().to(device)
train_and_evaluate(linear_model, "spectral_linear", run_dir, train_loader, val_loader, test_loader, device, epochs, learning_rate)

# Train Triadic Perceptron
triadic_model = TriadicPerceptron().to(device)
train_and_evaluate(triadic_model, "spectral_triadic", run_dir, train_loader, val_loader, test_loader, device, epochs, learning_rate)
