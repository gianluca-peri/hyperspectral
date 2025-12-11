import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from spectral.layers import SpectralTriadicOnly

def get_run_dir(base_dir="alpha_mnist"):
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

class TriadicPerceptron(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        # Register alpha as a buffer so it persists in state_dict
        self.register_buffer('alpha', torch.tensor(alpha))

        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("Alpha must be in the range [0, 1]")

        self.flatten = nn.Flatten()
        # MNIST images are 28x28 = 784
        self.triadic_pass = SpectralTriadicOnly(784, 10)
        self.binary_pass = nn.Linear(784, 10)
    def forward(self, x):
        x = self.flatten(x)
        triadic_out = self.triadic_pass(x)
        binary_out = self.binary_pass(x)
        return self.alpha * triadic_out + (1 - self.alpha) * binary_out


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
# Fix: Use larger batch size for speed (math handled below)
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
        "val_avg_confidence": [],
        "test_accuracy": 0.0,
        "test_loss": 0.0
    }
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_train_samples = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Fix: Weighted sum for accurate average
            running_loss += loss.item() * images.size(0)
            total_train_samples += images.size(0)
        
        avg_train_loss = running_loss / total_train_samples
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        total_confidence = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Fix: Weighted sum
                val_loss += loss.item() * images.size(0)
                
                # Calculate probabilities using softmax
                probs = torch.softmax(outputs, dim=1)
                # Get the highest probability (confidence) and the predicted class
                confidence, predicted = torch.max(probs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_confidence += confidence.sum().item()
        
        avg_val_loss = val_loss / total
        val_acc = 100 * correct / total
        avg_val_confidence = total_confidence / total
        
        print(f"[{model_name}] Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Avg Conf: {avg_val_confidence:.4f}")
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_avg_confidence"].append(avg_val_confidence)
        
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
            # Fix: Weighted sum
            test_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / total
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

alphas = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0]

for alpha in alphas:
    model_name = f"triadic_perceptron_alpha_{alpha}"
    model = TriadicPerceptron(alpha=alpha).to(device)
    train_and_evaluate(model, model_name, run_dir, train_loader, val_loader, test_loader, device, epochs, learning_rate)