from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from tqdm import trange
from contextlib import contextmanager
import torch
from torchvision.transforms.functional import to_pil_image
import lantern

from .model import Model


@contextmanager
def temporary_samples(model: Model, n_samples=512, batch_size=128, n_evaluations=100):
    try:
        with TemporaryDirectory() as temporary_dir, torch.random.fork_rng(), lantern.module_eval(
            model
        ):
            torch.manual_seed(13)
            for _ in trange(n_samples // batch_size, desc="generating samples"):
                for images in model.elucidated_sample(batch_size, n_evaluations):
                    pass

                for pil_image in pil_images(images):
                    pil_image.save(Path(temporary_dir) / f"{uuid4()}.png")
            yield temporary_dir
    finally:
        pass


def pil_images(images):
    return [to_pil_image(image) for image in images.clamp(0, 1)]
