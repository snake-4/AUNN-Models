import torch
import torch.nn as nn


class InvertedBottleneckBlock(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.0):
        super(InvertedBottleneckBlock, self).__init__()
        self.fn = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
        )
        if dropout != 0.0:
            self.fn.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(dim * expansion_factor, dim),
                    nn.Dropout(dropout),
                ]
            )
        else:
            self.fn.append(nn.Linear(dim * expansion_factor, dim))

        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.fn(self.ln(x))


class InvertedBottleneckMLP(nn.Module):
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
        self.outLayer = nn.Sequential(nn.Linear(hidden_width, out_dim), nn.Sigmoid())

        self.hiddenLayers = nn.ModuleList()
        for _ in range(hidden_depth):
            self.hiddenLayers.append(
                module=InvertedBottleneckBlock(hidden_width, expansion_factor, dropout)
            )

    def forward(self, x: torch.Tensor):
        x = self.inLayer(x)
        for layer in self.hiddenLayers:
            x = layer(x)
        return self.outLayer(x)


class NormalMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_width, hidden_depth, is_residual):
        super().__init__()

        self.isResidual = is_residual
        self.inLayer = nn.Linear(in_dim, hidden_width)
        self.outLayer = nn.Sequential(nn.Linear(hidden_width, out_dim), nn.Sigmoid())

        self.hiddenLayers = nn.ModuleList()
        for _ in range(hidden_depth):
            self.hiddenLayers.append(
                module=nn.Sequential(nn.Linear(hidden_width, hidden_width), nn.ReLU())
            )

    def forward(self, x: torch.Tensor):
        x = self.inLayer(x)

        lastX = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        for layer in self.hiddenLayers:
            x = layer(x)
            if self.isResidual:
                x = x + lastX
                lastX = x

        return self.outLayer(x)
