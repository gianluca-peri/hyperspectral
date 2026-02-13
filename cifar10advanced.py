import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from spectral.layers import SpectralTriadic


def get_run_dir(base_dir="cifar10advanced"):
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


# ---------------------------------------------------------------------------
# Direct model: Standard MLP-Mixer (non-triadic baseline)
# ---------------------------------------------------------------------------
class MixerBlock(nn.Module):
    """
    One standard MLP-Mixer block:
      1. Token mixing   – two-layer MLP across the patch (token) dimension
      2. Channel mixing – two-layer MLP across the channel dimension
    Both branches use pre-LayerNorm and a skip connection.
    """
    def __init__(self, num_patches, hidden_dim,
                 tokens_mlp_dim=None, channels_mlp_dim=None):
        super().__init__()
        if tokens_mlp_dim is None:
            tokens_mlp_dim = num_patches * 4
        if channels_mlp_dim is None:
            channels_mlp_dim = hidden_dim * 4

        # Token mixing
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_patches, tokens_mlp_dim),
            nn.GELU(),
            nn.Linear(tokens_mlp_dim, num_patches),
        )
        # Channel mixing
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, channels_mlp_dim),
            nn.GELU(),
            nn.Linear(channels_mlp_dim, hidden_dim),
        )

    def forward(self, x):
        # x: (B, num_patches, hidden_dim)
        # --- Token mixing ---
        residual = x
        y = self.token_norm(x).transpose(1, 2)   # (B, C, P)
        y = self.token_mlp(y)                     # (B, C, P)
        x = residual + y.transpose(1, 2)

        # --- Channel mixing ---
        residual = x
        y = self.channel_norm(x)                  # (B, P, C)
        y = self.channel_mlp(y)                   # (B, P, C)
        x = residual + y

        return x


class StandardMLPMixer(nn.Module):
    """
    Standard MLP-Mixer (Tolstikhin et al., 2021) using conventional
    two-layer MLPs for both token and channel mixing.
    Same overall architecture as SpectralMLPMixer so the comparison
    isolates the effect of the mixing layer type.
    """
    def __init__(self, in_channels=3, img_size=32, patch_size=4,
                 hidden_dim=128, num_layers=6, num_classes=10,
                 tokens_mlp_dim=None, channels_mlp_dim=None):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.hidden_dim = hidden_dim

        # Patch embedding (linear projection)
        self.patch_embed = nn.Linear(self.patch_dim, hidden_dim)

        # Mixer blocks
        self.blocks = nn.Sequential(*[
            MixerBlock(self.num_patches, hidden_dim,
                       tokens_mlp_dim, channels_mlp_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size

        # Extract non-overlapping patches -> (B, num_patches, patch_dim)
        x = x.unfold(2, P, P).unfold(3, P, P)             # (B, C, H//P, W//P, P, P)
        x = x.contiguous().permute(0, 2, 3, 1, 4, 5)      # (B, H//P, W//P, C, P, P)
        x = x.reshape(B, self.num_patches, self.patch_dim)

        x = self.patch_embed(x)     # (B, num_patches, hidden_dim)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)           # Global average pool over patches
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# Spectral model: MLP-Mixer with SpectralTriadic layers
# ---------------------------------------------------------------------------
class SpectralMixerBlock(nn.Module):
    """
    One Mixer block that performs:
      1. Token mixing   – SpectralTriadic applied across the patch (token) dimension
      2. Channel mixing – SpectralTriadic applied across the channel dimension
    Both branches use pre-LayerNorm and a skip connection.
    """
    def __init__(self, num_patches, hidden_dim):
        super().__init__()
        # Token mixing
        self.token_norm = nn.LayerNorm(num_patches)
        self.token_triadic = SpectralTriadic(num_patches, num_patches, bias=True)
        # Channel mixing
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_triadic = SpectralTriadic(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        # x: (B, num_patches, hidden_dim)
        B, P, C = x.shape

        # --- Token mixing ---
        residual = x
        y = x.transpose(1, 2)              # (B, C, P)
        y = self.token_norm(y)
        y = y.reshape(B * C, P)
        y = self.token_triadic(y)
        y = y.reshape(B, C, P).transpose(1, 2)
        x = residual + y

        # --- Channel mixing ---
        residual = x
        y = self.channel_norm(x)
        y = y.reshape(B * P, C)
        y = self.channel_triadic(y)
        y = y.reshape(B, P, C)
        x = residual + y

        return x


class SpectralMLPMixer(nn.Module):
    """
    MLP-Mixer whose token-mixing and channel-mixing MLPs are replaced
    by SpectralTriadic layers (linear + bilinear interaction in spectral
    parameterisation).
    """
    def __init__(self, in_channels=3, img_size=32, patch_size=4,
                 hidden_dim=128, num_layers=6, num_classes=10):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.hidden_dim = hidden_dim

        # Patch embedding (linear projection)
        self.patch_embed = nn.Linear(self.patch_dim, hidden_dim)

        # Mixer blocks
        self.blocks = nn.Sequential(*[
            SpectralMixerBlock(self.num_patches, hidden_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size

        # Extract non-overlapping patches -> (B, num_patches, patch_dim)
        x = x.unfold(2, P, P).unfold(3, P, P)             # (B, C, H//P, W//P, P, P)
        x = x.contiguous().permute(0, 2, 3, 1, 4, 5)      # (B, H//P, W//P, C, P, P)
        x = x.reshape(B, self.num_patches, self.patch_dim)

        x = self.patch_embed(x)     # (B, num_patches, hidden_dim)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)           # Global average pool over patches
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# Training / evaluation loop (same structure as cifar10.py)
# ---------------------------------------------------------------------------
def train_and_evaluate(model, model_name, run_dir, train_loader, val_loader,
                       test_loader, device, epochs, learning_rate,
                       weighted_decay=1e-2, cut_short=True):
    print(f"\nStarting training for {model_name}")
    model_dir = os.path.join(run_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                            weight_decay=weighted_decay)
    if cut_short:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
    else:
        scheduler = None
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

                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_confidence += confidence.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        if cut_short:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= 10:
                print(f"[{model_name}] Early stopping: no val loss improvement "
                      f"for {epochs_since_improvement} epochs.")
                break
        current_lr = optimizer.param_groups[0]['lr']

        val_acc = 100 * correct / total
        avg_val_confidence = total_confidence / total

        print(f"[{model_name}] Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%, "
              f"Val Avg Conf: {avg_val_confidence:.4f}, "
              f"LR: {current_lr:.6f}")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_avg_confidence"].append(avg_val_confidence)

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

    print(f"[{model_name}] Test Loss: {avg_test_loss:.4f}, "
          f"Test Acc: {test_acc:.2f}%")

    history["test_loss"] = avg_test_loss
    history["test_accuracy"] = test_acc

    with open(os.path.join(model_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    torch.save(model.state_dict(), os.path.join(model_dir, "model_final.pth"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Hyperparameters
    batch_size = 128
    learning_rate = 1e-2
    weight_decay = 1e-3
    epochs = 100
    # When True: use LR scheduler and allow early stopping (current behavior)
    # When False: fixed LR and train for all epochs
    cut_short = False

    num_workers = 4
    pin_memory = torch.cuda.is_available()
    print(f"DataLoader workers: {num_workers}, pin_memory: {pin_memory}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loading for CIFAR-10 with standard augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    full_train_raw = datasets.CIFAR10(root='./data', train=True, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                    transform=test_transform)

    train_size = 40000
    val_size = 10000
    train_subset, val_subset = random_split(full_train_raw,
                                            [train_size, val_size])

    train_dataset = torch.utils.data.Subset(
        datasets.CIFAR10(root='./data', train=True, download=False,
                         transform=train_transform),
        train_subset.indices
    )
    val_dataset = torch.utils.data.Subset(
        datasets.CIFAR10(root='./data', train=True, download=False,
                         transform=test_transform),
        val_subset.indices
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory)

    run_dir = get_run_dir()
    print(f"Saving results to {run_dir}")

    # Train Standard MLP-Mixer (non-triadic baseline)
    std_mixer = StandardMLPMixer(
        in_channels=3, img_size=32, patch_size=4,
        hidden_dim=128, num_layers=6, num_classes=10
    ).to(device)
    print(f"StandardMLPMixer params: "
          f"{sum(p.numel() for p in std_mixer.parameters()):,}")
    train_and_evaluate(std_mixer, "standard_mlp_mixer", run_dir,
                       train_loader, val_loader, test_loader, device,
                       epochs, learning_rate, weight_decay, cut_short)

    # Train SpectralTriadic MLP-Mixer
    spectral_mixer = SpectralMLPMixer(
        in_channels=3, img_size=32, patch_size=4,
        hidden_dim=128, num_layers=6, num_classes=10
    ).to(device)
    print(f"SpectralMLPMixer params: "
          f"{sum(p.numel() for p in spectral_mixer.parameters()):,}")
    train_and_evaluate(spectral_mixer, "spectral_triadic_mixer", run_dir,
                       train_loader, val_loader, test_loader, device,
                       epochs, learning_rate, weight_decay, cut_short)


if __name__ == '__main__':
    main()
