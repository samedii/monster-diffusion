import imgaug.augmenters as iaa
from datastream import Datastream

from monster_diffusion import settings
from .datasets import datasets
from .nonleaky_augmenter import NonleakyAugmenter


def train_datastream(dataset=None):
    if dataset is None:
        dataset = datasets()["train"]

    # augmenter_ = augmenter()
    nonleaky_augmenter = NonleakyAugmenter()
    # crop = iaa.CropToFixedSize(
    #     settings.INPUT_WIDTH,
    #     settings.INPUT_HEIGHT,
    # )
    # pad = iaa.PadToFixedSize(
    #     settings.INPUT_WIDTH,
    #     settings.INPUT_HEIGHT,
    #     position="uniform",
    # )
    return Datastream(dataset).map(
        lambda example: example.nonleaky_augment(nonleaky_augmenter)
    )


def augmenter():
    return iaa.Sequential(
        [
            iaa.HorizontalFlip(0.5),
            iaa.TranslateX(px=[0, 1, 2], cval=255),
            iaa.TranslateY(px=[0, 1, 2], cval=255),
            # iaa.Sometimes(0.25, iaa.Multiply((0.9, 1.1))),
            # iaa.Sometimes(0.7, iaa.Crop(percent=(0, 0.1))),
            # iaa.Sometimes(0.5, iaa.Resize({"height": (1, 1.2), "width": (1, 1.3)})),
            # iaa.Sometimes(0.5, iaa.RandAugment()),
            # iaa.Sometimes(0.5, iaa.TranslateX(percent=(-0.2, 0.2))),
            # iaa.Sometimes(0.5, affine),
            # iaa.Sometimes(0.5, shear_x),
            # iaa.Sometimes(0.5, shear_y),
            # iaa.Sometimes(0.5, iaa.AverageBlur(k=((1, 4), (1, 4)))),
        ],
        random_order=True,
    )
