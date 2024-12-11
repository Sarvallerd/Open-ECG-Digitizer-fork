import torch
from torch import nn


class RandomShiftTextTransform(nn.Module):
    def __init__(self) -> None:
        super(RandomShiftTextTransform, self).__init__()

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> None:
        pass
