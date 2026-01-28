import torch
from spectral.layers import SpectralTriadic

class MLP(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim=2, output_dim=1):
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

class TriadicMLP(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim=2, output_dim=1):
        super(TriadicMLP, self).__init__()
        self.layer1 = SpectralTriadic(input_dim, hidden_dim)
        self.layer2 = SpectralTriadic(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x