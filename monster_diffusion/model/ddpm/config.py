import numpy as np
from types import SimpleNamespace

from monster_diffusion import settings


config = SimpleNamespace(
    data=SimpleNamespace(
        dataset="CIFAR10",
        image_size=96,
        random_flip=True,
        centered=True,
        uniform_dequantization=False,
        num_channels=settings.INPUT_CHANNELS + settings.PRIOR_SHAPE[0],
    ),
    model=SimpleNamespace(
        name="ddpm",
        scale_by_sigma=False,
        ema_rate=0.9999,
        normalization="GroupNorm",
        nonlinearity="swish",
        nf=128,
        conditioning_channels=np.prod(settings.PRIOR_SHAPE),
        out_channels=settings.INPUT_CHANNELS,
        ch_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        resamp_with_conv=True,
        conditional=True,
        fir=False,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=True,
        resblock_type="biggan",
        progressive="none",
        progressive_input="none",
        progressive_combine="sum",
        attention_type="ddpm",
        init_scale=0.0,
        embedding_type="fourier",
        fourier_scale=16,
        conv_size=3,
        dropout=0.1,
    ),
)
