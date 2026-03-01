import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from spectral.layers import SpectralTriadic


def get_run_dir(base_dir="cifar100_vit"):
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
# Direct model: Standard Vision Transformer (Baseline)
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Standard linear projections for Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class StandardViT(nn.Module):
    def __init__(self, in_channels=3, img_size=32, patch_size=4,
                 hidden_dim=128, num_layers=6, num_heads=4, mlp_ratio=4.0, num_classes=100):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Linear(self.patch_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))

        self.blocks = nn.Sequential(*[
            TransformerBlock(hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size

        # Extract non-overlapping patches
        x = x.unfold(2, P, P).unfold(3, P, P)
        x = x.contiguous().permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(B, self.num_patches, self.patch_dim)

        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)
        
        # Use CLS token for classification
        cls_out = x[:, 0]
        x = self.head(cls_out)
        return x


# ---------------------------------------------------------------------------
# Spectral model: ViT with SpectralTriadic layers substituted 'in toto'
# ---------------------------------------------------------------------------

class SpectralTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        hidden_dim = int(dim * mlp_ratio)
        
        # Crucial modification: Replace MLP linears with SpectralTriadic and REMOVE non-linearities
        self.mlp_fc1 = SpectralTriadic(dim, hidden_dim, bias=True)
        self.mlp_fc2 = SpectralTriadic(hidden_dim, dim, bias=True)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        
        # --- MLP completely lacking non-linearities ---
        residual = x
        x_norm = self.norm2(x)
        B, N, C = x_norm.shape
        
        x_flat = x_norm.reshape(B * N, C)
        x_flat = self.mlp_fc1(x_flat)
        # Directly into the second layer (No GELU/ReLU)
        x_flat = self.mlp_fc2(x_flat)
        x_mlp = x_flat.reshape(B, N, C)
        
        x = residual + x_mlp
        return x


class SpectralViT(nn.Module):
    def __init__(self, in_channels=3, img_size=32, patch_size=4,
                 hidden_dim=128, num_layers=6, num_heads=4, mlp_ratio=4.0, num_classes=100):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        # Unmodified Patch Embedding using nn.Linear
        self.patch_embed = nn.Linear(self.patch_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))

        self.blocks = nn.Sequential(*[
            SpectralTransformerBlock(hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size

        # Extract non-overlapping patches
        x = x.unfold(2, P, P).unfold(3, P, P)
        x = x.contiguous().permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(B, self.num_patches, self.patch_dim)
        
        # Patch Embed (Standard Linear)
        x_embed = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x_embed), dim=1)
        x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)
        
        # Classification Head (B, hidden_dim -> already 2D)
        cls_out = x[:, 0]
        x = self.head(cls_out)
        return x


# ---------------------------------------------------------------------------
# Training / evaluation loop
# ---------------------------------------------------------------------------
def train_and_evaluate(model, model_name, run_dir, train_loader, val_loader,
                       test_loader, device, epochs, learning_rate,
                       weight_decay=0.05, use_cosine_annealing=True):
    print(f"\nStarting training for {model_name}")
    model_dir = os.path.join(run_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                            weight_decay=weight_decay)
    
    # Cosine Annealing is much more stable for ViT training
    if use_cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_avg_confidence": [],
        "test_accuracy": 0.0,
        "test_loss": 0.0
    }

    for epoch in range(epochs):
        epoch_start = time.time()
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
            scheduler.step()
            
        epoch_end = time.time()
        epoch_dur = epoch_end - epoch_start
        mins = int(epoch_dur // 60)
        secs = int(epoch_dur % 60)

        current_lr = optimizer.param_groups[0]['lr']
        val_acc = 100 * correct / total
        avg_val_confidence = total_confidence / total

        print(f"[{model_name}] Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%, "
              f"Val Avg Conf: {avg_val_confidence:.4f}, "
              f"LR: {current_lr:.6f}, "
              f"Epoch Time: {mins}m {secs}s")

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
    # Adjusted Hyperparameters suitable for Vision Transformers
    batch_size = 32
    learning_rate = 1e-3  # Lowered from 1e-2 for ViT stability
    weight_decay = 0.05   # Increased for better ViT regularization
    epochs = 100

    num_workers = 4
    pin_memory = torch.cuda.is_available()
    print(f"DataLoader workers: {num_workers}, pin_memory: {pin_memory}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loading for CIFAR-100
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

    full_train_raw = datasets.CIFAR100(root='./data', train=True, download=True)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True,
                                     transform=test_transform)

    train_size = 40000
    val_size = 10000
    train_subset, val_subset = random_split(full_train_raw,
                                            [train_size, val_size])

    train_dataset = torch.utils.data.Subset(
        datasets.CIFAR100(root='./data', train=True, download=False,
                          transform=train_transform),
        train_subset.indices
    )
    val_dataset = torch.utils.data.Subset(
        datasets.CIFAR100(root='./data', train=True, download=False,
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

    # Train Spectral Vision Transformer
    spectral_vit = SpectralViT(
        in_channels=3, img_size=32, patch_size=4,
        hidden_dim=128, num_layers=6, num_heads=4, mlp_ratio=3.5, num_classes=100
    ).to(device)
    print(f"SpectralViT params: "
          f"{sum(p.numel() for p in spectral_vit.parameters()):,}")
    train_and_evaluate(spectral_vit, "spectral_triadic_vit", run_dir,
                       train_loader, val_loader, test_loader, device,
                       epochs, learning_rate, weight_decay)

    # Train Standard Vision Transformer (non-spectral baseline)
    std_vit = StandardViT(
        in_channels=3, img_size=32, patch_size=4,
        hidden_dim=128, num_layers=6, num_heads=4, mlp_ratio=3.5, num_classes=100
    ).to(device)
    print(f"StandardViT params: "
          f"{sum(p.numel() for p in std_vit.parameters()):,}")
    train_and_evaluate(std_vit, "standard_vit", run_dir,
                       train_loader, val_loader, test_loader, device,
                       epochs, learning_rate, weight_decay)




if __name__ == '__main__':
    main()