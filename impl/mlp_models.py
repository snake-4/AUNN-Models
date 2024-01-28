import torch
import torch.nn as nn


class ResIBBlock(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.0):
        super().__init__()
        self.fn = nn.Sequential(nn.Linear(dim, dim * expansion_factor), nn.GELU())
        if dropout != 0.0:
            self.fn.append(nn.Dropout(dropout))
        self.fn.append(nn.Linear(dim * expansion_factor, dim))
        if dropout != 0.0:
            self.fn.append(nn.Dropout(dropout))

        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.fn(self.ln(x))


class ResIBModel(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_width,
        hidden_depth,
        expansion_factor=4,
        dropout=0.0,
    ):
        super().__init__()
        self.inLayer = nn.Linear(in_dim, hidden_width)
        self.outLayer = nn.Linear(hidden_width, out_dim)
        self.hiddenLayers = nn.ModuleList(
            ResIBBlock(hidden_width, expansion_factor, dropout)
            for _ in range(hidden_depth)
        )

    def forward(self, x: torch.Tensor):
        x = self.inLayer(x)
        for layer in self.hiddenLayers:
            x = layer(x)
        return self.outLayer(x)


class ResMLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fn = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.fn(self.ln(x))


class ResMLPModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_width, hidden_depth):
        super().__init__()
        self.inLayer = nn.Linear(in_dim, hidden_width)
        self.outLayer = nn.Linear(hidden_width, out_dim)
        self.hiddenLayers = nn.ModuleList(
            ResMLPBlock(hidden_width) for _ in range(hidden_depth)
        )

    def forward(self, x: torch.Tensor):
        x = self.inLayer(x)
        for layer in self.hiddenLayers:
            x = layer(x)
        return self.outLayer(x)
