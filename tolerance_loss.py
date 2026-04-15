import torch
import torch.nn as nn

class MarginToleranceLoss(nn.Module):
    def __init__(self, tol=1e-2, reduction='mean'):
        super().__init__()
        self.tol = tol
        self.reduction = reduction

    def forward(self, pred, y):
        loss = torch.relu(torch.abs(pred - y) - self.tol)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss  # 'none'