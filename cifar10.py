import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from spectral.layers import SpectralTriadic

def get_run_dir(base_dir="cifar10"):
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

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # CIFAR-10 images are 3x32x32 = 3072
        self.relu = nn.ReLU()
        self.layernorm1 = nn.LayerNorm(256)
        self.layernorm2 = nn.LayerNorm(64)
        self.layer1 = nn.Linear(3072, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layernorm1(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layernorm2(x)
        x = self.layer3(x)
        return x

class TriadicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layernorm1 = nn.LayerNorm(256)
        self.layernorm2 = nn.LayerNorm(64)
        self.layer1 = SpectralTriadic(3072, 256)
        self.layer2 = SpectralTriadic(256, 64)
        self.layer3 = SpectralTriadic(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layernorm1(x)
        x = self.layer2(x)
        x = self.layernorm2(x)
        x = self.layer3(x)
        return x


def train_and_evaluate(model, model_name, run_dir, train_loader, val_loader, test_loader, device, epochs, learning_rate, weighted_decay=1e-2):
    print(f"\nStarting training for {model_name}")
    model_dir = os.path.join(run_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weighted_decay)
    # LR scheduler: halve LR if no val loss improvement for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    
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
        total_confidence = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate probabilities using softmax
                probs = torch.softmax(outputs, dim=1)
                # Get the highest probability (confidence) and the predicted class
                confidence, predicted = torch.max(probs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_confidence += confidence.sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        # Scheduler step and early stopping logic
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        # Stop training if no improvement for 10 epochs
        if epochs_since_improvement >= 10:
            print(f"[{model_name}] Early stopping: no val loss improvement for {epochs_since_improvement} epochs.")
            break
        current_lr = optimizer.param_groups[0]['lr']
        
        val_acc = 100 * correct / total
        avg_val_confidence = total_confidence / total
        
        print(f"[{model_name}] Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Avg Conf: {avg_val_confidence:.4f}, LR: {current_lr:.6f}")
        
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

def main():
    # Hyperparameters
    batch_size = 128
    learning_rate = 1e-2
    weight_decay = 1e-3
    epochs = 200

    # DataLoader workers (adjust for faster loading)
    num_workers = 4
    pin_memory = True if torch.cuda.is_available() else False
    print(f"DataLoader workers: {num_workers}, pin_memory: {pin_memory}")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loading for CIFAR-10 with standard augmentation: random crop + horizontal flip
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Load raw train dataset (no transform) to split indices, then create train/val subsets
    full_train_raw = datasets.CIFAR10(root='./data', train=True, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # Split train into train and validation (40000 train, 10000 val)
    train_size = 40000
    val_size = 10000
    train_subset, val_subset = random_split(full_train_raw, [train_size, val_size])

    # Create datasets with proper transforms using the split indices
    train_dataset = torch.utils.data.Subset(
        datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform),
        train_subset.indices
    )
    val_dataset = torch.utils.data.Subset(
        datasets.CIFAR10(root='./data', train=True, download=False, transform=test_transform),
        val_subset.indices
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Setup save directory
    run_dir = get_run_dir()
    print(f"Saving results to {run_dir}")

    # Train Triadic Perceptron
    triadic_model = TriadicMLP().to(device)
    train_and_evaluate(triadic_model, "spectral triadic mlp", run_dir, train_loader, val_loader, test_loader, device, epochs, learning_rate, weight_decay)

    # Train Linear Perceptron
    direct_model = MLP().to(device)
    train_and_evaluate(direct_model, "direct mlp", run_dir, train_loader, val_loader, test_loader, device, epochs, learning_rate, weight_decay)


if __name__ == '__main__':
    main()
