import torch
from spectral.layers import SpectralTriadic, SpectralTriadicOnly

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

class AugmentedTriadicMLP(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim=1):
        super(AugmentedTriadicMLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.triadic1 = SpectralTriadicOnly(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
        self.triadic2 = SpectralTriadicOnly(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x)) + self.triadic1(x)
        x = self.linear2(x) + self.triadic2(x)
        return x

class DeepTriadicMLP(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim=1, num_layers=3):
        super(DeepTriadicMLP, self).__init__()
        layers = [SpectralTriadic(input_dim, hidden_dim)]
        for _ in range(num_layers - 3):
            layers.append(SpectralTriadic(hidden_dim, hidden_dim))
        layers.append(SpectralTriadic(hidden_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

import torch

class DeepTriadicMLP_Funnel(torch.nn.Module):
    # Must be deeply tested, it is recomended to use DeepTriadicMLP in general
    def __init__(self, hidden_dim, input_dim, output_dim=1, num_layers=3):
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2 (input->...->output)"
        hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)
        n_hidden_layers = num_layers - 1
        hidden_dims = []
        start = hidden_dim - (n_hidden_layers - 1)
        for i in range(n_hidden_layers):
            hidden_dims.append(max(start + i, 2))
        layers = []
        layers.append(SpectralTriadic(input_dim, hidden_dims[0]))
        for in_d, out_d in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(SpectralTriadic(in_d, out_d))
        layers.append(SpectralTriadic(hidden_dims[-1], output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
