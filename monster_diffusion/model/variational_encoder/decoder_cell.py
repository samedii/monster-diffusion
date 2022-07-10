from torch import nn

from .swish import Swish
from .squeeze_excitation import SqueezeExcitation


class DecoderCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        expanded_channels = channels * 6
        self.seq = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            Swish(),
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=5,
                padding=2,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            Swish(),
            nn.Conv2d(expanded_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            SqueezeExcitation(channels),
        )

    def forward(self, x):
        return x + self.seq(x)
