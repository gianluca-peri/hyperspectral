import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from lib.layers import SpectralTriadic
from lib.utils import get_run_dir
from lib.utils import save_history, save_final_test_evaluation, save_model
from lib.train_and_evaluate import train_and_evaluate

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Setup save directory
run_dir = get_run_dir("cifar10advanced")

# Choice of criterion
criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------------------------
# Direct model: Standard MLP-Mixer (non-triadic baseline)
# ---------------------------------------------------------------------------
class MixerBlock(nn.Module):
    """
    One standard MLP-Mixer block:
      1. Token mixing   - two-layer MLP across the patch (token) dimension
      2. Channel mixing - two-layer MLP across the channel dimension
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
      1. Token mixing   - SpectralTriadic applied across the patch (token) dimension
      2. Channel mixing - SpectralTriadic applied across the channel dimension
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

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

models = [
    (
        StandardMLPMixer(
            in_channels=3, img_size=32, patch_size=4,
            hidden_dim=128, num_layers=6, num_classes=10
        ).to(device),
        "standard_mlp_mixer",
    ),
    (
        SpectralMLPMixer(
            in_channels=3, img_size=32, patch_size=4,
            hidden_dim=128, num_layers=6, num_classes=10
        ).to(device),
        "spectral_triadic_mixer",
    ),
]

learning_rate = 1e-2
weight_decay = 1e-3
epochs = 100
for model, name in models:
    print(f"{model.__class__.__name__} params: {sum(p.numel() for p in model.parameters()):,}")

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
        verbose=True,
    )

    save_history(run_dir, name, history)
    save_final_test_evaluation(run_dir, name, final_test_evaluation)
    save_model(run_dir, name, model)
