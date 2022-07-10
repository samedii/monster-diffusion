from torch import nn

from .swish import Swish
from .squeeze_excitation import SqueezeExcitation


class EncoderCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.BatchNorm2d(dim),
            Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            SqueezeExcitation(dim),
        )

    def forward(self, x):
        return x + self.seq(x)
