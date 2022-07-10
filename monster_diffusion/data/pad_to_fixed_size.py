import imgaug

from monster_diffusion import settings


pad_to_fixed_size = imgaug.augmenters.PadToFixedSize(
    width=settings.INPUT_WIDTH,
    height=settings.INPUT_HEIGHT,
    position="center",
    pad_cval=255,
)
