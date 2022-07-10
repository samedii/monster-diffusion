import math
import copy
from typing import List
from tqdm import tqdm
from functools import partial, wraps
from contextlib import contextmanager
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.special import expm1
import torchvision.transforms as T

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from einops_exts import rearrange_many, repeat_many, check_shape
from einops_exts.torch import EinopsToAndFrom

from .resize_right import resize


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)

    return inner


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(val, length=1):
    if isinstance(val, list):
        val = tuple(val)

    return val if isinstance(val, tuple) else ((val,) * length)


def module_device(module):
    return next(module.parameters()).device


@contextmanager
def null_context(*args, **kwargs):
    yield


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def pad_tuple_to_length(t, length, fillvalue=None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fillvalue,) * remain_length))


# tensor helpers


def log(t, eps: float = 1e-12):
    return torch.log(t.clamp(min=eps))


def l2norm(t):
    return F.normalize(t, dim=-1)


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def masked_mean(t, *, dim, mask=None):
    if not exists(mask):
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)


def resize_image_to(image, target_image_size):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    scale_factors = target_image_size / orig_image_size
    return resize(image, scale_factors=scale_factors)


# image normalization functions
# ddpms expect images to be in the range of -1 to 1


def normalize_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5


# classifier free guidance functions


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# attention pooling


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), nn.LayerNorm(dim)
        )

    def forward(self, x, latents, mask=None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        # attention

        sim = einsum("... i d, ... j d  -> ... i j", q, k)

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        num_latents=64,
        num_latents_mean_pooled=4,  # number of latents derived from mean pooled representation of the sequence
        max_seq_len=512,
        ff_mult=4,
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.to_latents_from_mean_pooled_seq = None

        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, mask=None):
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device=device))

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, "n d -> b n d", b=x.shape[0])

        if exists(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = masked_mean(
                x,
                dim=1,
                mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool),
            )
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask=mask) + latents
            latents = ff(latents) + latents

        return latents


# attention


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head=64,
        heads=8,
        causal=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim)
        )

    def forward(self, x, mask=None, attn_bias=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        q = q * self.scale

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), "d -> b 1 d", b=b)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # calculate query / key similarities

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# decoder


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(x, "i -> i 1") * rearrange(emb, "j -> 1 j")
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, norm=True):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else nn.Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x, scale_shift=None):
        x = self.groupnorm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, cond_dim=None, time_cond_dim=None, groups=8):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        if exists(cond_dim):
            self.cross_attn = EinopsToAndFrom(
                "b c h w",
                "b (h w) c",
                CrossAttention(dim=dim_out, context_dim=cond_dim),
            )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, cond=None, time_emb=None):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x)

        if exists(self.cross_attn):
            assert exists(cond)
            h = self.cross_attn(h, context=cond) + h

        h = self.block2(h, scale_shift=scale_shift)

        return h + self.res_conv(x)


class CrossAttention(nn.Module):
    def __init__(
        self, dim, *, context_dim=None, dim_head=64, heads=8, norm_context=False
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim)
        )

    def forward(self, x, context, mask=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(
            self.null_kv.unbind(dim=-2), "d -> b h 1 d", h=self.heads, b=b
        )

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head=32, heads=8, dropout=0.05):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.SiLU()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias=False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias=False, padding=1, groups=inner_dim),
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias=False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias=False, padding=1, groups=inner_dim),
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias=False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias=False, padding=1, groups=inner_dim),
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1, bias=False), ChanLayerNorm(dim)
        )

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = rearrange_many((q, k, v), "b (h c) x y -> (b h) (x y) c", h=h)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        context = einsum("b n d, b n e -> b d e", k, v)
        out = einsum("b n d, b d e -> b n e", q, context)
        out = rearrange(out, "(b h) (x y) d -> b (h d) x y", h=h, x=x, y=y)

        out = self.nonlin(out)
        return self.to_out(out)


def FeedForward(dim, mult=2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias=False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias=False),
    )


def ChanFeedForward(
    dim, mult=2
):  # in paper, it seems for self attention layers they did feedforwards with twice channel width
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv2d(dim, hidden_dim, 1, bias=False),
        nn.GELU(),
        ChanLayerNorm(hidden_dim),
        nn.Conv2d(hidden_dim, dim, 1, bias=False),
    )


class TransformerBlock(nn.Module):
    def __init__(self, dim, *, heads=8, dim_head=32, ff_mult=2):
        super().__init__()
        self.attn = EinopsToAndFrom(
            "b c h w", "b (h w) c", Attention(dim=dim, heads=heads, dim_head=dim_head)
        )
        self.ff = ChanFeedForward(dim=dim, mult=ff_mult)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class CrossEmbedLayer(nn.Module):
    def __init__(self, dim_in, kernel_sizes, dim_out=None, stride=2):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                nn.Conv2d(
                    dim_in,
                    dim_scale,
                    kernel,
                    stride=stride,
                    padding=(kernel - stride) // 2,
                )
            )

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)


class Unet(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_embed_dim=1024,
        text_embed_dim=0,  # get_encoded_dim(DEFAULT_T5_NAME),
        num_resnet_blocks=1,
        cond_dim=None,
        num_image_tokens=4,
        num_time_tokens=2,
        fourier_embed_time_or_noise=True,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        channels_out=None,
        attn_dim_head=64,
        attn_heads=8,
        ff_mult=2.0,
        lowres_cond=False,  # for cascading diffusion - https://cascaded-diffusion.github.io/
        layer_attns=True,
        attend_at_middle=True,  # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        layer_cross_attns=True,
        use_linear_attn=False,
        cond_on_text=True,
        max_text_len=256,
        init_dim=None,
        init_conv_kernel_size=7,
        resnet_groups=8,
        init_cross_embed_kernel_sizes=(3, 7, 15),
        cross_embed_downsample=False,
        cross_embed_downsample_kernel_sizes=(2, 4),
        attn_pool_text=True,
        attn_pool_num_latents=32,
        dropout=0.0,
    ):
        super().__init__()
        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        self._locals.pop("self", None)
        self._locals.pop("__class__", None)

        # for eventual cascading diffusion

        self.lowres_cond = lowres_cond

        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        init_channels = (
            channels if not lowres_cond else channels * 2
        )  # in cascading diffusion, one concats the low resolution image, blurred, for conditioning the higher resolution synthesis
        init_dim = default(init_dim, dim)

        self.init_conv = CrossEmbedLayer(
            init_channels,
            dim_out=init_dim,
            kernel_sizes=init_cross_embed_kernel_sizes,
            stride=1,
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time, image embeddings, and optional text encoding

        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4

        # embedding time for discrete gaussian diffusion or log(snr) noise for continuous version

        self.fourier_embed_time_or_noise = fourier_embed_time_or_noise

        if fourier_embed_time_or_noise:
            self.to_time_hiddens = nn.Sequential(
                SinusoidalPosEmb(dim), nn.Linear(dim, time_cond_dim), nn.SiLU()
            )
        else:
            self.to_time_hiddens = nn.Sequential(
                Rearrange("... -> ... 1"),
                nn.Linear(1, time_cond_dim),
                nn.SiLU(),
                nn.LayerNorm(time_cond_dim),
                nn.Linear(time_cond_dim, time_cond_dim),
                nn.SiLU(),
                nn.LayerNorm(time_cond_dim),
            )

        self.to_lowres_time_hiddens = None
        if lowres_cond:
            self.to_lowres_time_hiddens = copy.deepcopy(self.to_time_hiddens)
            time_cond_dim *= 2

        # project to time tokens as well as time hiddens

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange("b (r d) -> b r d", r=num_time_tokens),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(time_cond_dim, time_cond_dim))

        self.norm_cond = nn.LayerNorm(cond_dim)
        self.norm_mid_cond = nn.LayerNorm(cond_dim)

        # text encoding conditioning (optional)

        self.text_to_cond = None

        if cond_on_text:
            assert exists(
                text_embed_dim
            ), "text_embed_dim must be given to the unet if cond_on_text is True"
            self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)

        # finer control over whether to condition on text encodings

        self.cond_on_text = cond_on_text

        # attention pooling

        self.attn_pool = (
            PerceiverResampler(
                dim=cond_dim,
                depth=2,
                dim_head=attn_dim_head,
                heads=attn_heads,
                num_latents=attn_pool_num_latents,
            )
            if attn_pool_text
            else None
        )

        # for classifier free guidance

        self.null_image_embed = nn.Parameter(torch.randn(1, num_image_tokens, cond_dim))

        self.max_text_len = max_text_len
        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))

        # attention related params

        attn_kwargs = dict(heads=attn_heads, dim_head=attn_dim_head)

        num_layers = len(in_out)

        # resnet block klass

        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_layers)
        resnet_groups = cast_tuple(resnet_groups, num_layers)

        layer_attns = cast_tuple(layer_attns, num_layers)
        layer_cross_attns = cast_tuple(layer_cross_attns, num_layers)

        assert all(
            [
                layers == num_layers
                for layers in list(
                    map(len, (resnet_groups, layer_attns, layer_cross_attns))
                )
            ]
        )

        # downsample klass

        downsample_klass = Downsample
        if cross_embed_downsample:
            downsample_klass = partial(
                CrossEmbedLayer, kernel_sizes=cross_embed_downsample_kernel_sizes
            )

        # whether to use linear attention or not for layers where normal attention is computationally prohibitive

        full_attn_substitute = (
            partial(
                LinearAttention,
                heads=attn_heads,
                dim_head=attn_dim_head,
                dropout=dropout,
            )
            if use_linear_attn
            else nn.Identity
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        layer_params = [
            num_resnet_blocks,
            resnet_groups,
            layer_attns,
            layer_cross_attns,
        ]
        reversed_layer_params = list(map(reversed, layer_params))

        for ind, (
            (dim_in, dim_out),
            layer_num_resnet_blocks,
            groups,
            layer_attn,
            layer_cross_attn,
        ) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)
            layer_cond_dim = cond_dim if layer_cross_attn else None

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_in,
                            dim_out,
                            cond_dim=layer_cond_dim,
                            time_cond_dim=time_cond_dim,
                            groups=groups,
                        ),
                        nn.ModuleList(
                            [
                                ResnetBlock(dim_out, dim_out, groups=groups)
                                for _ in range(layer_num_resnet_blocks)
                            ]
                        ),
                        TransformerBlock(
                            dim=dim_out,
                            heads=attn_heads,
                            dim_head=attn_dim_head,
                            ff_mult=ff_mult,
                        )
                        if layer_attn
                        else full_attn_substitute(dim_out),
                        downsample_klass(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(
            mid_dim,
            mid_dim,
            cond_dim=cond_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups[-1],
        )
        self.mid_attn = (
            EinopsToAndFrom(
                "b c h w", "b (h w) c", Residual(Attention(mid_dim, **attn_kwargs))
            )
            if attend_at_middle
            else None
        )
        self.mid_block2 = ResnetBlock(
            mid_dim,
            mid_dim,
            cond_dim=cond_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups[-1],
        )

        for ind, (
            (dim_in, dim_out),
            layer_num_resnet_blocks,
            groups,
            layer_attn,
            layer_cross_attn,
        ) in enumerate(zip(reversed(in_out[1:]), *reversed_layer_params)):
            layer_cond_dim = cond_dim if layer_cross_attn else None

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_out * 2,
                            dim_in,
                            cond_dim=layer_cond_dim,
                            time_cond_dim=time_cond_dim,
                            groups=groups,
                        ),
                        nn.ModuleList(
                            [
                                ResnetBlock(dim_in, dim_in, groups=groups)
                                for _ in range(layer_num_resnet_blocks)
                            ]
                        ),
                        TransformerBlock(
                            dim=dim_in,
                            heads=attn_heads,
                            dim_head=attn_dim_head,
                            ff_mult=ff_mult,
                        )
                        if layer_attn
                        else full_attn_substitute(dim_in),
                        Upsample(dim_in),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            ResnetBlock(dim, dim, groups=resnet_groups[0]),
            nn.Conv2d(dim, self.channels_out, 1),
        )

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        text_embed_dim,
        channels,
        channels_out,
        cond_on_text,
        fourier_embed_time_or_noise,
    ):
        if (
            lowres_cond == self.lowres_cond
            and channels == self.channels
            and cond_on_text == self.cond_on_text
            and text_embed_dim == self._locals["text_embed_dim"]
            and fourier_embed_time_or_noise == self.fourier_embed_time_or_noise
            and channels_out == self.channels_out
        ):
            return self

        updated_kwargs = dict(
            lowres_cond=lowres_cond,
            text_embed_dim=text_embed_dim,
            channels=channels,
            channels_out=channels_out,
            cond_on_text=cond_on_text,
            fourier_embed_time_or_noise=fourier_embed_time_or_noise,
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        *,
        lowres_cond_img=None,
        lowres_noise_times=None,
        text_embeds=None,
        text_mask=None,
        cond_drop_prob=0.0,
    ):
        batch_size, device = x.shape[0], x.device

        # add low resolution conditioning, if present

        assert not (
            self.lowres_cond and not exists(lowres_cond_img)
        ), "low resolution conditioning image must be present"
        assert not (
            self.lowres_cond and not exists(lowres_noise_times)
        ), "low resolution conditioning noise time must be present"

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim=1)

        # initial convolution

        x = self.init_conv(x)

        # time conditioning

        time_hiddens = self.to_time_hiddens(time)

        # add the time conditioning for the noised lowres conditioning, if needed

        if exists(self.to_lowres_time_hiddens):
            lowres_time_hiddens = self.to_lowres_time_hiddens(lowres_noise_times)
            time_hiddens = torch.cat((time_hiddens, lowres_time_hiddens), dim=-1)

        # derive time tokens

        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)

        # conditional dropout

        text_keep_mask = prob_mask_like(
            (batch_size,), 1 - cond_drop_prob, device=device
        )

        text_keep_mask = rearrange(text_keep_mask, "b -> b 1 1")

        # take care of text encodings (optional)

        text_tokens = None

        if exists(text_embeds) and self.cond_on_text:
            text_tokens = self.text_to_cond(text_embeds)

            text_tokens = text_tokens[:, : self.max_text_len]

            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len

            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))

            if exists(text_mask):
                if remainder > 0:
                    text_mask = F.pad(text_mask, (0, remainder), value=False)

                text_mask = rearrange(text_mask, "b n -> b n 1")
                text_keep_mask = text_mask & text_keep_mask

            null_text_embed = self.null_text_embed.to(
                text_tokens.dtype
            )  # for some reason pytorch AMP not working

            text_tokens = torch.where(text_keep_mask, text_tokens, null_text_embed)

            if exists(self.attn_pool):
                text_tokens = self.attn_pool(text_tokens)

        # main conditioning tokens (c)

        c = (
            time_tokens
            if not exists(text_tokens)
            else torch.cat((time_tokens, text_tokens), dim=-2)
        )

        # normalize conditioning tokens

        c = self.norm_cond(c)

        # go through the layers of the unet, down and up

        hiddens = []

        for init_block, resnet_blocks, attn_block, downsample in self.downs:
            x = init_block(x, c, t)

            for resnet_block in resnet_blocks:
                x = resnet_block(x)

            x = attn_block(x)

            hiddens.append(x)
            x = downsample(x)

        x = self.mid_block1(x, c, t)

        if exists(self.mid_attn):
            x = self.mid_attn(x)

        x = self.mid_block2(x, c, t)

        for init_block, resnet_blocks, attn_block, upsample in self.ups:
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = init_block(x, c, t)

            for resnet_block in resnet_blocks:
                x = resnet_block(x)

            x = attn_block(x)
            x = upsample(x)

        return self.final_conv(x)


# predefined unets, with configs lining up with hyperparameters in appendix of paper


class BaseUnet64(Unet):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            dict(
                dim=512,
                dim_mults=(1, 2, 3, 4),
                num_resnet_blocks=3,
                layer_attns=(False, True, True, True),
                layer_cross_attns=(False, True, True, True),
                attn_heads=8,
                ff_mult=2.0,
            )
        )
        super().__init__(*args, **kwargs)


class SRUnet256(Unet):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            dict(
                dim=128,
                dim_mults=(1, 2, 4, 8),
                num_resnet_blocks=(2, 4, 8, 8),
                layer_attns=(False, False, False, True),
                layer_cross_attns=(False, False, False, True),
                attn_heads=8,
                ff_mult=2.0,
            )
        )
        super().__init__(*args, **kwargs)


class SRUnet1024(Unet):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            dict(
                dim=128,
                dim_mults=(1, 2, 4, 8),
                num_resnet_blocks=(2, 4, 8, 8),
                layer_attns=False,
                layer_cross_attns=(False, False, False, True),
                attn_heads=8,
                ff_mult=2.0,
            )
        )
        super().__init__(*args, **kwargs)


class ImagenUnet(Unet):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            dict(
                channels=1,
                dim=64,
                dim_mults=(1, 2, 3, 4),
                num_resnet_blocks=3,
                layer_attns=(False, True, True, True),
                layer_cross_attns=(False, True, True, True),
                attn_heads=8,
                ff_mult=2.0,
                max_text_len=1,
            )
        )
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x,
        time,
        embeds,
        *_,
        **kwargs,
    ):
        kwargs.update(dict(text_embeds=embeds[:, None]))
        return super().forward(x, time, **kwargs)
