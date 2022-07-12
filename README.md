# monster diffusion

Generates 48x48 images of monsters.
![generated monsters](docs/samples.png)

| FID@20 | FID@100 |
| ------ | ------- |
| 52.5   | 41.8    |

FID between `train` and `early_stopping` is **7.36**. `FID@k` means that the
samples require `k` model evaluations.

Inspired by:

- [DiVAE : Photorealistic Images Synthesis with Denoising Diffusion Decoder](https://arxiv.org/pdf/2206.00386.pdf)
- [DiffuseVAE: Efficient, Controllable and High-Fidelity Generation from Low-Dimensional Latents](https://github.com/kpandey008/DiffuseVAE)
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
- [Katherine Crowson's k-diffusion](https://github.com/crowsonkb/k-diffusion)
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://github.com/yang-song/score_sde_pytorch)
- [Velocity diffusion sampling](https://github.com/crowsonkb/v-diffusion-pytorch)
- [Progressive Distillation for Fast Sampling of Diffusion Models](https://openreview.net/forum?id=TIdIXIpzhoI)
- [Diffusion Models Beat GANS on Image Synthesis](https://github.com/crowsonkb/guided-diffusion)

## Install

```bash
poetry add git+https://github.com/samedii/monster-diffusion.git
```

## Usage

```python
import monster_diffusion

image = monster_diffusion.sample()
```

## Development

### Installation

Setup environment:

```bash
poetry install
```

If you need cuda 11:

```bash
poetry run pip uninstall torch torchvision -y && poetry run pip install torch==1.11.0 torchvision==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Debugging

Add a debug config for a python module in vscode, e.g.:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Module",
      "type": "python",
      "request": "launch",
      "module": "operations.debug.train",
      "justMyCode": true
    }
  ]
}
```

Also set a shortcut like `Shift+Enter` for `"Evaluate in debug console"` to interactively
run code you are trying to fix while debugging.

## Training

```bash
guild run train
guild run retrain model=<model-hash>
guild run evaluate model=<model-hash>
guild tensorboard <model-hash>
```
