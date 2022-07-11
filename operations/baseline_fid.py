from tempfile import TemporaryDirectory
from pathlib import Path
from uuid import uuid4
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from monster_diffusion import data
import nicefid

from monster_diffusion import settings


if __name__ == "__main__":
    evaluate_datastreams = data.evaluate_datastreams()
    evaluate_data_loaders = {
        name: (
            (
                datastream.data_loader(
                    batch_size=16,
                    collate_fn=list,
                    num_workers=8,
                )
            )
        )
        for name, datastream in evaluate_datastreams.items()
    }

    def generator(data_loader):
        for examples in data_loader:
            yield torch.stack(
                [TF.to_tensor(example.image).div(255) for example in examples]
            )

    fid_score = nicefid.compute_fid(
        nicefid.Features.from_iterator(generator(evaluate_data_loaders["train"])),
        nicefid.Features.from_iterator(
            generator(evaluate_data_loaders["early_stopping"])
        ),
    )
    print(f"fid score between train and early stopping: {fid_score}")
