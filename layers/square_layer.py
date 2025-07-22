import torch
import torch.nn as nn


class SquareLayer(nn.Module):
    def __init__(self):
        super(SquareLayer, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([inputs, inputs**2], dim=-1)
