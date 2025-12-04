import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, train_eigenvectors=True):
        super(SpectralLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.train_eigenvectors = train_eigenvectors
        
        if self.bias:
            # Add bias neuron to input features
            self.phi = nn.Parameter(torch.Tensor(out_features, in_features + 1), requires_grad=train_eigenvectors)
            self.l_in = nn.Parameter(torch.Tensor(in_features + 1))
        else:
            self.phi = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=train_eigenvectors)
            self.l_in = nn.Parameter(torch.Tensor(in_features))
        
        self.l_out = nn.Parameter(torch.Tensor(out_features))
            
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.xavier_uniform_(self.phi)
        nn.init.uniform_(self.l_in, -1, 1)
        nn.init.uniform_(self.l_out, -1, 1)

    def forward(self, input):
        # Weight matrix W = phi * diag(l_in) - diag(l_out) * phi
        # So w_ij = (l_j - l_i) * phi_ij

        if self.bias:
            ones = torch.ones(input.size(0), 1, device=input.device)
            input = torch.cat([input, ones], dim=1)
        
        weight = (self.l_in.unsqueeze(0) - self.l_out.unsqueeze(1)) * self.phi
        return F.linear(input, weight, bias=None)

class SpectralTriadic(nn.Module):
    def __init__(self, in_features, out_features, bias=True, train_eigenvectors=True, train_triadic_eigenvectors=False):
        super(SpectralTriadic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.train_eigenvectors = train_eigenvectors
        
        if self.bias:
            # Add bias neuron to input features
            self.phi = nn.Parameter(torch.Tensor(out_features, in_features + 1), requires_grad=train_eigenvectors)
            self.l_in = nn.Parameter(torch.Tensor(in_features + 1, in_features + 1))
        else:
            self.phi = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=train_eigenvectors)
            self.l_in = nn.Parameter(torch.Tensor(in_features, in_features))
        
        self.l_out = nn.Parameter(torch.Tensor(out_features))
        self.phi_triadic = nn.Parameter(torch.Tensor(out_features, in_features, in_features), requires_grad=train_triadic_eigenvectors)
        self.l_in_triadic = nn.Parameter(torch.Tensor(in_features, in_features))
        self.l_out_triadic = nn.Parameter(torch.Tensor(out_features))
            
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.xavier_uniform_(self.phi)
        nn.init.uniform_(self.l_in, -1, 1)
        nn.init.uniform_(self.l_out, -1, 1)
        nn.init.xavier_uniform_(self.phi_triadic.view(self.out_features, -1))
        nn.init.uniform_(self.l_in_triadic, -1, 1)
        nn.init.uniform_(self.l_out_triadic, -1, 1)

    def forward(self, input):
        if self.bias:
            ones = torch.ones(input.size(0), 1, device=input.device)
            input = torch.cat([input, ones], dim=1)
        
        # Linear weight
        # w_kj = (l_jj - l_k) * phi_kj
        l_in_diag = torch.diagonal(self.l_in)
        weight_linear = (l_in_diag.unsqueeze(0) - self.l_out.unsqueeze(1)) * self.phi
        out_linear = F.linear(input, weight_linear, bias=None)
        
        # Triadic weight
        # w_kij = (L_ij - L_k) * phi_kij
        weight_triadic = (self.l_in_triadic.unsqueeze(0) - self.l_out_triadic.unsqueeze(1).unsqueeze(2)) * self.phi_triadic
        
        # Compute bilinear term: sum_ij w_kij * x_i * x_j
        if self.bias:
            # Exclude bias neuron from triadic computation
            input = input[:, :-1]
        out_triadic = torch.einsum('bi,bj,kij->bk', input, input, weight_triadic)
        
        return out_linear + out_triadic
