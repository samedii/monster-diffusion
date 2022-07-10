import torch
from torch import nn
from lantern import module_device, Tensor

from .variational_features import VariationalFeatures
from .ldm import Encoder
from monster_diffusion.model import standardize
from monster_diffusion import settings


class VariationalEncoderLDM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

    @property
    def device(self):
        return module_device(self)

    @staticmethod
    def train_sample(mean, log_variance):
        std = torch.exp(0.5 * log_variance)
        return mean + torch.randn_like(std) * std

    @staticmethod
    def kl(mean, log_variance):
        loss = -0.5 * (1 + log_variance - mean**2 - torch.exp(log_variance))
        return loss.flatten(start_dim=1).mean(dim=1)

    def forward(
        self,
        images: Tensor.dims("NCHW").shape(-1, *settings.INPUT_SHAPE).float(),
        noise: Tensor.dims("NCHW").shape(-1, *settings.INPUT_SHAPE).float(),
    ):
        denoised_xs = standardize.encode(images).to(self.device)

        mean, log_variance = self.encoder(
            torch.cat([denoised_xs, noise.to(self.device)], dim=1)
        ).chunk(2, dim=1)

        return VariationalFeatures(
            features=self.train_sample(mean, log_variance),
            kl=self.kl(mean, log_variance),
        )

    def features_(
        self,
        images: Tensor.dims("NCHW").shape(-1, *settings.INPUT_SHAPE).uint8(),
        noise: Tensor.dims("NCHW").shape(-1, *settings.INPUT_SHAPE).float(),
    ):
        return self.forward(images, noise)

    def features(
        self,
        images: Tensor.dims("NCHW").shape(-1, *settings.INPUT_SHAPE).uint8(),
        noise: Tensor.dims("NCHW").shape(-1, *settings.INPUT_SHAPE).float(),
    ):
        if self.training:
            raise Exception(
                "Cannot run features method while in training mode. Use features_"
            )
        return self.features_(images, noise)
