import torch.nn as nn

class HybridMSELoss(nn.Module):
    def __init__(self, alpha=1e-2, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        abs_mse = diff.pow(2)
        rel_mse = abs_mse / (target.abs() + self.eps)
        return ((1 - self.alpha) * abs_mse + self.alpha * rel_mse).mean()
