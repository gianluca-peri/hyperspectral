import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from lib.layers import SpectralTriadic
from lib.utils import get_run_dir
from lib.utils import save_history, save_final_test_evaluation, save_model
from lib.utils import compute_dataset_mean_std
from lib.train_and_evaluate import train_and_evaluate

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Setup save directory
run_dir = get_run_dir("cifar10")

# Choice of criterion
criterion = nn.CrossEntropyLoss()

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

# Data loading for CIFAR-10 with standard augmentation: random crop + horizontal flip
stats_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
mean, std = compute_dataset_mean_std(stats_dataset)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
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

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

models = [
    (TriadicMLP().to(device), "spectral_triadic_mlp"),
    (MLP().to(device), "direct_mlp")
]

learning_rate = 1e-2
weight_decay = 1e-3
epochs = 100
for model, name in models:
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history, final_test_evaluation, model = train_and_evaluate(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        epochs,
        device,
        scheduler_plateau=scheduler_plateau,
        verbose=True
    )

    save_history(run_dir, name, history)
    save_final_test_evaluation(run_dir, name, final_test_evaluation)
    save_model(run_dir, name, model)
