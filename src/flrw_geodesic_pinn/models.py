"""Neural trial functions for the three representations in the paper."""
from __future__ import annotations
import torch
from torch import nn


def build_mlp(in_dim=1, out_dim=2, width=64, depth=5):
    layers = [nn.Linear(in_dim, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers += [nn.Linear(width, out_dim)]
    net = nn.Sequential(*layers)
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    return net


def derivative(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]


class AffineSecondOrderPINN(nn.Module):
    def __init__(self, eta0, x0, eta0p, x0p, lambda0, lambda_max, width=64, depth=5, init_state=None):
        super().__init__()
        self.net = build_mlp(1, 2, width, depth)
        if init_state is not None:
            self.net.load_state_dict(init_state)
        self.eta0, self.x0 = eta0, x0
        self.eta0p, self.x0p = eta0p, x0p
        self.lambda0, self.lambda_max = lambda0, lambda_max

    def forward(self, lam):
        s = 2.0 * (lam - self.lambda0) / (self.lambda_max - self.lambda0) - 1.0
        raw = self.net(s)
        d = lam - self.lambda0
        return self.eta0 + self.eta0p*d + d**2*raw[:, :1], self.x0 + self.x0p*d + d**2*raw[:, 1:2]


class ConformalFirstIntegralPINN(nn.Module):
    def __init__(self, eta0, x0, dx0, dlambda0, eta_min, eta_max, lambda0=0.0, width=64, depth=5, init_state=None):
        super().__init__()
        self.net = build_mlp(1, 2, width, depth)
        if init_state is not None:
            self.net.load_state_dict(init_state)
        self.eta0, self.x0 = eta0, x0
        self.dx0, self.dlambda0 = dx0, dlambda0
        self.eta_min, self.eta_max = eta_min, eta_max
        self.lambda0 = lambda0

    def forward(self, eta):
        s = 2.0 * (eta - self.eta_min) / (self.eta_max - self.eta_min) - 1.0
        raw = self.net(s)
        h = eta - self.eta0
        return self.x0 + self.dx0*h + h**2*raw[:, :1], self.lambda0 + self.dlambda0*h + h**2*raw[:, 1:2]
