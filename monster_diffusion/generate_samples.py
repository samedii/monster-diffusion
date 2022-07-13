from tqdm import trange
import torch
import lantern

from .model.model import Model


def generate_samples(model: Model, n_samples=2048, batch_size=32, n_evaluations=100):
    with lantern.module_eval(model):
        torch.manual_seed(18)
        for _ in trange(n_samples // batch_size, desc="generating samples"):
            for images in model.elucidated_sample(batch_size, n_evaluations):
                pass

            yield images.clamp(0, 1)
