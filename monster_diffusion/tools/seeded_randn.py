import torch


def seeded_randn(shape, seed):
    with torch.random.fork_rng([]):  # does not fork cuda rng
        torch.random.manual_seed(seed)
        return torch.randn(shape)
