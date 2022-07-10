from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from lantern import FunctionalBase, Tensor
from typing import Optional

from monster_diffusion import settings


class VariationalFeatures(FunctionalBase):
    features: Tensor.shape(-1, *settings.PRIOR_SHAPE)
    kl: Optional[Tensor.dims("N")]

    def __len__(self):
        return len(self.denoised_images)

    @property
    def device(self):
        return self.denoised_xs.device

    def to(self, device):
        return self.replace(
            features=self.features.to(device),
            kl=None if self.kl is None else self.kl.to(device),
        )

    def losses(self):
        return self.kl

    def loss(self):
        return self.kl.mean()

    @staticmethod
    def sample(size, prior_std=1.0):
        return VariationalFeatures(
            features=torch.randn((size, *settings.PRIOR_SHAPE)) * prior_std
        )
