from tqdm import trange
import torch
import lantern

from .model import Model


def generate_samples(model: Model, n_samples=1024, batch_size=32, n_evaluations=100):
    with lantern.module_eval(model):
        torch.manual_seed(13)
        for _ in trange(n_samples // batch_size, desc="generating samples"):
            for images in model.sample(batch_size, n_evaluations):
                pass

            yield images.clamp(0, 1)
