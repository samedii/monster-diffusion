from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from lantern import FunctionalBase, Tensor

from monster_diffusion import settings
from . import standardize, diffusion


class Prediction(FunctionalBase):
    denoised_image: Tensor.dims("CHW").shape(*settings.INPUT_SHAPE)
    diffused_image: Tensor.dims("CHW").shape(*settings.INPUT_SHAPE)
    t: Tensor.dims("").float()

    def representation(self, example):
        denoised_image = self.denoised_image.detach().clamp(0, 1) * 255
        diffused_image = self.diffused_image.detach().clamp(0, 1) * 255
        horizontal_line = torch.full((3, 2, denoised_image.shape[-1]), 255)
        return np.uint8(
            torch.cat(
                [
                    torch.from_numpy(example.image).clamp(0, 255).permute(2, 0, 1),
                    horizontal_line,
                    diffused_image,
                    horizontal_line,
                    denoised_image,
                ],
                dim=-2,
            )
            .permute(1, 2, 0)
            .numpy()
        )

    def pil_image(self, example):
        return Image.fromarray(self.representation(example))


class PredictionBatch(FunctionalBase):
    denoised_xs: Tensor.dims("NCHW").shape(-1, *settings.INPUT_SHAPE).float()
    diffused_images: Tensor.dims("NCHW").shape(-1, *settings.INPUT_SHAPE).float()
    ts: Tensor.dims("N").float()

    def __len__(self):
        return len(self.denoised_images)

    def __getitem__(self, index):
        return Prediction(
            denoised_image=self.denoised_images[index],
            diffused_image=self.diffused_images[index],
            t=self.ts[index],
        )

    @property
    def device(self):
        return self.denoised_xs.device

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def loss(self, images, noise):
        return self.weighted_image_mse(images)

    def weighted_image_mse(self, images):
        weights = (
            (self.from_sigmas**2 + diffusion.sigma_data**2)
            / (self.from_sigmas**2 * diffusion.sigma_data**2)
        ).view(-1)
        targets = self.targets(images)
        assert targets.shape == self.denoised_xs.shape
        assert weights.shape == (len(images),)
        return (
            (self.denoised_xs - targets)
            .square()
            .flatten(start_dim=1)
            .mean(dim=1)
            .mul(weights)
            .mean()
        )

    def image_mse(self, images):
        targets = self.targets(images)
        assert targets.shape == self.denoised_xs.shape
        return F.mse_loss(self.denoised_xs, targets)

    def eps_mse(self, noise):
        eps = self.eps
        assert eps.shape == noise.shape
        return F.mse_loss(eps, noise.to(self.device))

    def targets(self, images):
        return standardize.encode(images).to(self.device)

    @staticmethod
    def sigmas(ts):
        return ts[:, None, None, None]

    @staticmethod
    def alphas(ts):
        return torch.ones_like(PredictionBatch.sigmas(ts))

    @property
    def from_sigmas(self):
        return self.sigmas(self.ts).to(self.device)

    @property
    def from_alphas(self):
        return self.alphas(self.ts).to(self.device)

    @property
    def from_xs(self):
        return self.diffused_xs

    @property
    def diffused_xs(self):
        return standardize.encode(self.diffused_images).to(self.device)

    @property
    def denoised_images(self):
        return standardize.decode(self.denoised_xs).cpu()

    @property
    def eps(self):
        return (self.diffused_xs - self.denoised_xs) / self.from_sigmas

    def step(self, to_ts):
        """
        Step the diffused image forward to `to_t`. Decreasing the amount of noise
        by moving closer to the predicted denoised image.
        """
        to_ts = to_ts.to(self.device)
        to_alphas, to_sigmas = self.alphas(to_ts), self.sigmas(to_ts)

        # to_diffused_xs = self.denoised_xs * to_alphas + self.eps * to_sigmas
        # return standardize.decode(to_diffused_xs)

        to_diffused_xs = self.diffused_xs + self.eps * (to_sigmas - self.from_sigmas)
        return standardize.decode(to_diffused_xs)

    def correction(self, previous_diffused_images, previous_ts, previous_eps):
        previous_diffused_xs = standardize.encode(
            previous_diffused_images.to(self.device)
        )
        corrected_diffused_xs = (
            previous_diffused_xs
            + (self.from_sigmas - self.sigmas(previous_ts))
            * (self.eps + previous_eps)
            / 2
        )
        return standardize.decode(corrected_diffused_xs)
