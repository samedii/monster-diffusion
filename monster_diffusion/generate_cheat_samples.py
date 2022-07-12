from tqdm import tqdm
import torch
import torchvision.transforms.functional as TF
import lantern

from .tools.seeded_randn import seeded_randn
from .model.model import Model
from monster_diffusion import settings


def generate_cheat_samples(
    model: Model, variational_encoder, data_loader, n_evaluations=100
):
    torch.manual_seed(13)
    for examples in tqdm(data_loader, desc="generating cheat samples"):
        images = torch.stack([TF.to_tensor(example.image) for example in examples]).div(
            255
        )
        noise = torch.stack(
            [
                seeded_randn(settings.INPUT_SHAPE, abs(example.hash()))
                for example in examples
            ]
        )

        with lantern.module_eval(variational_encoder):
            variational_features = variational_encoder.features(images, noise)

        ts = model.schedule_ts(n_evaluations // 2)[0].repeat(len(examples))
        diffused_images = model.diffuse(images, ts, noise)
        with lantern.module_eval(model):
            for images in model.sample(
                len(images),
                variational_features=variational_features,
                diffused_images=diffused_images,
                n_evaluations=n_evaluations,
            ):
                pass

            yield images.clamp(0, 1)
