import torch
import torch.nn as nn


class PseudoLinearSampler(nn.Module):
    def __init__(self, steps):
        """
        Faster sampling by keeping track of previous eps and
        using a pseudo-linear step. Slightly modified from original
        where first eps were collected with prk steps.
        """
        super().__init__()
        self.plms_eps_queue = nn.Parameter(None, requires_grad=False)
        self.steps = nn.Parameter(steps[:, None], requires_grad=False)
        self.called_eps = 0

    def __iter__(self):
        steps = self.steps.clone()
        for index, (from_value, to_value) in enumerate(zip(steps, steps[1:])):
            if self.called_eps != index:
                raise ValueError("Noise was not updated with sampler.eps_")
            yield index, from_value, to_value

    def __len__(self):
        return len(self.steps) - 1

    def eps_(self, eps):
        self.called_eps += 1
        if len(self.plms_eps_queue) < 3:
            return self.ddim_eps_(eps)
        else:
            return self.plms_eps_(eps)

    def plms_eps_(self, eps):
        eps_prime = (
            55 * eps
            - 59 * self.plms_eps_queue[-1]
            + 37 * self.plms_eps_queue[-2]
            - 9 * self.plms_eps_queue[-3]
        ) / 24
        self.plms_eps_queue = nn.Parameter(
            torch.cat([self.plms_eps_queue[1:], eps.detach()[None]]),
            requires_grad=False,
        )
        return eps_prime

    def ddim_eps_(self, eps):
        self.plms_eps_queue = nn.Parameter(
            torch.cat([self.plms_eps_queue, eps.detach()[None]]), requires_grad=False
        )
        return eps
