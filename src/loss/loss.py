from torch import nn
from torch.nn import functional as F
import torch


class DiceLoss(nn.Module):
    def __init__(self, device: str | torch.device):
        super(DiceLoss, self).__init__()
        self.pos_class_weight = torch.tensor([1, 1, 5, 1.0]).to(device)[None, :, None, None]

    def forward(self, pred: torch.Tensor, target_one_hot: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)
        smooth = 1e-6
        intersection = torch.sum(pred * target_one_hot, dim=(0, 2, 3))
        union = torch.sum(pred, dim=(0, 2, 3)) + torch.sum(target_one_hot, dim=(0, 2, 3))
        dice: torch.Tensor = 1 - (2 * intersection + smooth) / (union + smooth)
        return dice.mean()
