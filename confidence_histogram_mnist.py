import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spectral.layers import SpectralTriadic, DirectSpaceTriadic

# Define the model classes (must match the ones used for training)
class DirectSpaceTriadicPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # MNIST images are 28x28 = 784
        self.layer = DirectSpaceTriadic(784, 10)

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

def get_confidences(model, dataloader, device):
    model.eval()
    confidences = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().numpy())
    return np.array(confidences)

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Paths
    run_dir = "mnist/run_0"
    direct_model_path = os.path.join(run_dir, "direct_space_triadic", "model_final.pth")
    spectral_model_path = os.path.join(run_dir, "spectral_triadic", "model_final.pth")

    # Load models
    print("Loading Direct Space Triadic model...")
    direct_model = DirectSpaceTriadicPerceptron().to(device)
    direct_model.load_state_dict(torch.load(direct_model_path, map_location=device))

    print("Loading Spectral Triadic model...")
    spectral_model = TriadicPerceptron().to(device)
    spectral_model.load_state_dict(torch.load(spectral_model_path, map_location=device))

    # Get confidences
    print("Calculating confidences for Direct Space Triadic...")
    direct_confidences = get_confidences(direct_model, test_loader, device)

    print("Calculating confidences for Spectral Triadic...")
    spectral_confidences = get_confidences(spectral_model, test_loader, device)

    # Load histories
    direct_history_path = os.path.join(run_dir, "direct_space_triadic", "history.json")
    spectral_history_path = os.path.join(run_dir, "spectral_triadic", "history.json")

    with open(direct_history_path, 'r') as f:
        direct_history = json.load(f)
    with open(spectral_history_path, 'r') as f:
        spectral_history = json.load(f)

    # Plot histogram
    plt.rc('font', size=16)
    print("Plotting...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Validation Confidence
    # Assuming both have same number of epochs
    epochs = range(1, len(direct_history["val_avg_confidence"]) + 1)
    ax1.plot(epochs, direct_history["val_avg_confidence"], label='Direct Space Triadic', marker='o')
    ax1.plot(epochs, spectral_history["val_avg_confidence"], label='Spectral Triadic', marker='^')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Confidence')
    ax1.set_title('Validation Confidence during Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histogram
    ax2.hist(direct_confidences, bins=50, alpha=0.5, label='Direct Space Triadic', density=True)
    ax2.hist(spectral_confidences, bins=50, alpha=0.5, label='Spectral Triadic', density=True)
    
    ax2.set_xlabel('Confidence (Max Softmax Probability)')
    ax2.set_ylabel('Density (Log Scale)')
    ax2.set_title('Confidence Histogram on MNIST Test Set')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    output_path = os.path.join("mnist", "confidence_histogram_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    # plt.show() # Uncomment if running in an environment with display

if __name__ == "__main__":
    main()
