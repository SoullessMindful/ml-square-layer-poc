import torch.nn as nn

class RelativeMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = (diff ** 2) / (target.abs() + self.eps)
        return loss.mean()
