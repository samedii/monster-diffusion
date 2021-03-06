from tqdm import tqdm
import numpy as np
from scipy import integrate
import torch
import torch.nn as nn
import torch.nn.functional as F
from lantern import module_device, Tensor

from monster_diffusion import model, settings
from . import standardize
from .unet import UNet
from .ddpm import DDPM
from .imagen import ImagenUnet
from . import k_diffusion
from .pseudo_linear_sampler import PseudoLinearSampler
from . import diffusion
from .variational_encoder import VariationalFeatures
from .variational_encoder.decoder_cell import DecoderCell
from .prediction import PredictionBatch


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.network = UNet(
        #     image_size=96,
        #     in_channels=1,
        #     model_channels=128,
        #     out_channels=1,
        #     num_res_blocks=2,
        #     attention_resolutions=[4, 8],  # how many times downsampled x2
        #     dropout=0.13,
        #     channel_mult=(1, 2, 2, 2),
        #     num_heads=4,
        #     use_scale_shift_norm=True,
        #     conditioning_dim=np.prod(settings.PRIOR_SHAPE),
        # )
        # self.network = DDPM()
        # self.network = ImagenUnet(text_embed_dim=np.prod(settings.PRIOR_SHAPE))
        self.network = k_diffusion.Model(
            mapping_cond_dim=np.prod(settings.PRIOR_SHAPE) + 9,
            unet_cond_dim=settings.PRIOR_SHAPE[0],
        )

        latent_channels = 64
        channels = 64
        self.decoded_sample = nn.Sequential(
            nn.Conv2d(latent_channels, channels, kernel_size=1, bias=False),
            DecoderCell(channels),
            DecoderCell(channels),
        )

    @property
    def device(self):
        return module_device(self)

    @staticmethod
    def training_ts(size):
        random_ts = (diffusion.P_mean + torch.randn(size) * diffusion.P_std).exp()
        return random_ts

    @staticmethod
    def schedule_ts(n_steps):
        indices = torch.arange(n_steps)
        return (
            diffusion.sigma_max ** (1 / diffusion.raw)
            + indices
            / (n_steps - 1)
            * (
                diffusion.sigma_min ** (1 / diffusion.raw)
                - diffusion.sigma_max ** (1 / diffusion.raw)
            )
        ) ** diffusion.raw

    @staticmethod
    def evaluation_ts():
        n_steps = 1000
        schedule_ts = Model.schedule_ts(n_steps)
        return torch.cat(
            [
                schedule_ts,
                Model.reversed_ts(schedule_ts, n_steps),
            ]
        ).unique()

    @staticmethod
    def sigmas(ts):
        return PredictionBatch.sigmas(ts)

    @staticmethod
    def alphas(ts):
        return PredictionBatch.alphas(ts)

    @staticmethod
    def random_noise(size):
        return standardize.decode(
            torch.randn(size, *settings.INPUT_SHAPE)
            * Model.sigmas(Model.schedule_ts(100)[:1])
        )

    @staticmethod
    def diffuse(
        images: Tensor.dims("NCHW").shape(-1, *settings.INPUT_SHAPE).float(),
        ts,
        noise=None,
    ):
        x0 = standardize.encode(images)
        if isinstance(ts, float) or ts.ndim == 0:
            ts = torch.full((x0.shape[0],), ts)

        if noise is None:
            noise = torch.randn_like(x0)

        assert x0.shape == noise.shape

        return standardize.decode(x0 + noise * Model.sigmas(ts))

    def c_skip(self, ts):
        return diffusion.sigma_data**2 / (
            diffusion.sigma_data**2 + self.sigmas(ts) ** 2
        )

    def c_out(self, ts):
        return (
            self.sigmas(ts)
            * diffusion.sigma_data
            / torch.sqrt(diffusion.sigma_data**2 + self.sigmas(ts) ** 2)
        )

    def c_in(self, ts):
        return 1 / torch.sqrt(diffusion.sigma_data**2 + self.sigmas(ts) ** 2)

    def c_noise(self, ts):
        return 1 / 4 * self.sigmas(ts).log().view(-1)

    def denoised_(
        self,
        diffused_images: Tensor.dims("NCHW").shape(-1, *settings.INPUT_SHAPE).float(),
        ts: Tensor.dims("N"),
        nonleaky_augmentations: Tensor.dims("NK"),
        variational_features: VariationalFeatures,
    ) -> Tensor.dims("NCHW").shape(-1, *settings.INPUT_SHAPE):
        """
        Parameterization from https://arxiv.org/pdf/2206.00364.pdf
        """
        diffused_xs = standardize.encode(diffused_images.to(self.device))
        ts = ts.to(self.device)
        nonleaky_augmentations = nonleaky_augmentations.to(self.device)

        output = self.network(
            self.c_in(ts) * diffused_xs,
            self.c_noise(ts),
            mapping_cond=torch.cat(
                [
                    nonleaky_augmentations,
                    self.decoded_sample(variational_features.features).flatten(
                        start_dim=1
                    ),
                ],
                dim=1,
            ),
            unet_cond=F.interpolate(
                variational_features.features,
                size=diffused_xs.shape[2:],
                mode="nearest",
            ),
        )
        return self.c_skip(ts) * diffused_xs + self.c_out(ts) * output

    def forward(
        self,
        diffused_images: Tensor.dims("NCHW"),
        ts: Tensor.dims("N"),
        nonleaky_augmentations: Tensor.dims("NK"),
        variational_features: VariationalFeatures,
    ):
        denoised_xs = self.denoised_(
            diffused_images,
            ts,
            nonleaky_augmentations,
            variational_features,
        )
        return PredictionBatch(
            denoised_xs=denoised_xs,
            diffused_images=diffused_images,
            ts=ts,
        )

    def predictions_(
        self,
        diffused_images: Tensor.dims("NCHW"),
        ts: Tensor.dims("N"),
        nonleaky_augmentations: Tensor.dims("NK"),
        variational_features: VariationalFeatures,
    ):
        return self.forward(
            diffused_images,
            ts,
            nonleaky_augmentations,
            variational_features,
        )

    def predictions(
        self,
        diffused_images: Tensor.dims("NCHW"),
        ts: Tensor.dims("N"),
        nonleaky_augmentations: Tensor.dims("NK"),
        variational_features: VariationalFeatures,
    ):
        if self.training:
            raise Exception(
                "Cannot run predictions method while in training mode. Use predictions_"
            )
        return self.predictions_(
            diffused_images,
            ts,
            nonleaky_augmentations,
            variational_features,
        )

    @staticmethod
    def gamma(ts, n_steps):
        return torch.where(
            (ts >= diffusion.S_tmin) & (ts <= diffusion.S_tmax),
            torch.minimum(
                torch.tensor(diffusion.S_churn / n_steps), torch.tensor(2).sqrt() - 1
            ).to(ts),
            torch.zeros_like(ts),
        )

    @staticmethod
    def reversed_ts(ts, n_steps):
        return ts + Model.gamma(ts, n_steps) * ts

    def inject_noise(self, diffused_images, ts, reversed_ts):
        diffused_xs = standardize.encode(diffused_images).to(self.device)
        new_noise = torch.randn_like(diffused_xs) * diffusion.S_noise
        reversed_diffused_xs = (
            diffused_xs
            + (self.sigmas(reversed_ts).square() - self.sigmas(ts).square()).sqrt()
            * new_noise
        )
        return standardize.decode(reversed_diffused_xs)

    def sample(
        self,
        size,
        n_evaluations=100,
        progress=False,
        variational_features=None,
        diffused_images=None,
    ):
        return self.elucidated_sample(
            size,
            n_evaluations,
            progress,
            variational_features,
            diffused_images,
        )

    def elucidated_sample(
        self,
        size,
        n_evaluations=100,
        progress=False,
        variational_features=None,
        diffused_images=None,
    ):
        """
        Elucidated stochastic sampling from https://arxiv.org/pdf/2206.00364.pdf
        """
        if self.training:
            raise Exception("Cannot run sample method while in training mode.")
        if diffused_images is None:
            diffused_images = self.random_noise(size).to(self.device)
        if variational_features is None:
            variational_features = VariationalFeatures.sample(size).to(self.device)
        nonleaky_augmentations = torch.zeros(
            (size, 9), dtype=torch.float32, device=self.device
        )

        n_steps = n_evaluations // 2
        schedule_ts = self.schedule_ts(n_steps)[:, None].repeat(1, size).to(self.device)

        progress = tqdm(total=n_steps, disable=not progress, leave=False)
        for from_ts, to_ts in zip(schedule_ts[:-1], schedule_ts[1:]):
            reversed_ts = self.reversed_ts(from_ts, n_steps).clamp(max=schedule_ts[0])
            diffused_images = self.inject_noise(diffused_images, from_ts, reversed_ts)

            predictions = self.predictions(
                diffused_images,
                reversed_ts,
                nonleaky_augmentations,
                variational_features,
            )
            reversed_eps = predictions.eps
            reversed_diffused_images = diffused_images
            diffused_images = predictions.step(to_ts)

            predictions = self.predictions(
                diffused_images,
                to_ts,
                nonleaky_augmentations,
                variational_features,
            )
            diffused_images = predictions.correction(
                reversed_diffused_images, reversed_ts, reversed_eps
            )
            progress.update()
            yield predictions.denoised_images

        reversed_ts = self.reversed_ts(to_ts, n_steps)
        diffused_images = self.inject_noise(diffused_images, to_ts, reversed_ts)

        predictions = self.predictions(
            diffused_images,
            reversed_ts,
            nonleaky_augmentations,
            variational_features,
        )
        progress.close()
        yield predictions.denoised_images

    @staticmethod
    def linear_multistep_coeff(order, sigmas, from_index, to_index):
        if order - 1 > from_index:
            raise ValueError(f"Order {order} too high for step {from_index}")

        def fn(tau):
            prod = 1.0
            for k in range(order):
                if to_index == k:
                    continue
                prod *= (tau - sigmas[from_index - k]) / (
                    sigmas[from_index - to_index] - sigmas[from_index - k]
                )
            return prod

        return integrate.quad(
            fn, sigmas[from_index], sigmas[from_index + 1], epsrel=1e-4
        )[0]

    def linear_multistep_sample(
        self,
        size,
        n_evaluations=100,
        progress=False,
        variational_features=None,
        diffused_images=None,
        order=4,
    ):
        """
        Katherine Crowson's linear multistep method from https://github.com/crowsonkb/k-diffusion/blob/4fdb34081f7a09f16c33d3344a042e5bea8e69ee/k_diffusion/sampling.py
        """
        if self.training:
            raise Exception("Cannot run sample method while in training mode.")
        if diffused_images is None:
            diffused_images = self.random_noise(size)
        if variational_features is None:
            variational_features = VariationalFeatures.sample(size).to(self.device)
        nonleaky_augmentations = torch.zeros(
            (size, 9), dtype=torch.float32, device=self.device
        )
        diffused_images = diffused_images.to(self.device)

        n_steps = n_evaluations
        schedule_ts = self.schedule_ts(n_steps)[:, None].repeat(1, size).to(self.device)

        epses = list()
        progress = tqdm(total=n_steps, disable=not progress, leave=False)
        for from_index, from_ts, to_ts in zip(
            range(n_steps), schedule_ts[:-1], schedule_ts[1:]
        ):

            predictions = self.predictions(
                diffused_images,
                from_ts,
                nonleaky_augmentations,
                variational_features,
            )
            epses.append(predictions.eps)
            if len(epses) > order:
                epses.pop(0)

            current_order = len(epses)
            coeffs = [
                self.linear_multistep_coeff(
                    current_order,
                    self.sigmas(schedule_ts[:, 0]).cpu().flatten(),
                    from_index,
                    to_index,
                )
                for to_index in range(current_order)
            ]

            diffused_xs = standardize.encode(diffused_images)
            diffused_xs = diffused_xs + sum(
                coeff * eps for coeff, eps in zip(coeffs, reversed(epses))
            )
            diffused_images = standardize.decode(diffused_xs)

            progress.update()
            yield predictions.denoised_images

        predictions = self.predictions(
            diffused_images,
            to_ts,
            nonleaky_augmentations,
            variational_features,
        )
        progress.close()
        yield predictions.denoised_images

    def euler_sample(
        self,
        size,
        n_evaluations=100,
        n_correction_steps=0,
        target_snr=0.1,
        progress=False,
    ):
        diffused_images = self.random_noise(size)
        variational_features = VariationalFeatures.sample(size).to(self.device)

        n_steps = n_evaluations // (1 + n_correction_steps)
        ts = self.schedule_ts(n_steps).repeat(1, size).to(self.device)

        progress = tqdm(total=n_steps, disable=not progress, leave=False)
        for from_ts, to_ts in zip(ts[:-1], ts[1:]):

            for _ in range(n_correction_steps):
                predictions_corrector = self.predictions(
                    diffused_images,
                    from_ts,
                    variational_features,
                )
                diffused_images = predictions_corrector.langevin_correction(target_snr)

            reversed_ts = self.reversed_ts(from_ts, n_steps).clamp(
                max=diffusion.sigma_max
            )
            diffused_images = self.inject_noise(diffused_images, from_ts, reversed_ts)

            predictions = self.predictions(
                diffused_images,
                reversed_ts,
                variational_features,
            )
            diffused_images = predictions.step(to_ts)
            yield predictions.denoised_images
            progress.update()

        predictions = self.predictions(
            diffused_images,
            to_ts,
            variational_features,
        )
        progress.close()
        yield predictions.denoised_images

    def plms_sample(self, size, n_evaluations=100, progress=False):
        diffused_images = self.random_noise(size).to(self.device)
        variational_features = VariationalFeatures.sample(size).to(self.device)

        n_steps = n_evaluations
        sampler = PseudoLinearSampler(self.schedule_ts(n_steps)).to(self.device)
        progress = tqdm(total=n_steps, disable=not progress, leave=False)
        for _, from_ts, to_ts in sampler:
            predictions = self.predictions(
                diffused_images,
                from_ts,
                variational_features,
            )
            eps = sampler.eps_(predictions.eps)
            diffused_images = self.transfer(diffused_images, from_ts, to_ts, eps)
            progress.update()
            yield predictions.denoised_images

        predictions = self.predictions(
            diffused_images,
            to_ts,
            variational_features,
        )
        progress.close()
        yield predictions.denoised_images

    @staticmethod
    def transfer(diffused_images, from_ts, to_ts, eps):
        from_diffused_xs = standardize.encode(diffused_images)
        from_alphas, from_sigmas = Model.alphas(from_ts), Model.sigmas(from_ts)
        to_alphas, to_sigmas = Model.alphas(to_ts), Model.sigmas(to_ts)

        denoised_xs = (from_diffused_xs - eps * from_sigmas) / from_alphas
        to_diffused_xs = denoised_xs * to_alphas + eps * to_sigmas
        return standardize.decode(to_diffused_xs)


def test_model():
    import torch

    torch.set_grad_enabled(False)
    model = Model().eval()
    model.predictions_(
        torch.zeros((2, *settings.INPUT_SHAPE)),
        torch.zeros((2,)),
        VariationalFeatures.sample(2),
    )
