import numpy as np
import datastream
from monster_dataset.cropped.dataframe import dataframe

from monster_diffusion import settings
from .example import Example
from .pad_to_fixed_size import pad_to_fixed_size
from .replace_background import replace_background


def datasets(frozen=True):
    return (
        datastream.Dataset.from_dataframe(
            dataframe()[
                lambda df: (df.height <= settings.INPUT_HEIGHT)
                & (df.width <= settings.INPUT_WIDTH)
            ]
            .assign(stratify=lambda df: df["root_folder"])
            .astype(dict(stratify=str, root_folder=str, key=str))
        )
        .map(lambda row: Example.from_row(row))
        .map(
            lambda example: example.replace(
                image=replace_background(
                    example.image, np.full_like(example.image, 255)[..., :3]
                )
            ).augment(pad_to_fixed_size)
        )
        .split(
            key_column="key",
            proportions=dict(train=0.6, early_stopping=0.4),
            stratify_column="stratify",
            seed=700,
            filepath="monster_diffusion/splits/splits.json",
            frozen=frozen,
        )
    )
