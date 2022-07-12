import numpy as np


def replace_background(transparent, background):
    mask = np.float32(transparent[..., -1:] == 255)
    return np.uint8(transparent[..., :3] * mask + background * (1 - mask))
