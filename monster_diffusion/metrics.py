import torch
import lantern


def train_metrics():
    return dict(
        loss=lantern.Metric().reduce(lambda state, loss: dict(loss=loss.item())),
        variational_loss=lantern.Metric().reduce(
            lambda state, variational_loss: dict(
                variational_loss=variational_loss.item()
            )
        ),
        image_mse=lantern.Metric().reduce(
            lambda state, image_mse: dict(image_mse=image_mse.item())
        ),
        eps_mse=lantern.Metric().reduce(
            lambda state, eps_mse: dict(eps_mse=eps_mse.item())
        ),
    )


def evaluate_metrics():
    return dict(
        loss=lantern.Metric().aggregate(
            lambda losses: dict(loss=torch.stack(losses).mean().item())
        ),
        variational_loss=lantern.Metric().aggregate(
            lambda variational_losses: dict(variational_loss=torch.stack(variational_losses).mean().item())
        ),
        image_mse=lantern.Metric().aggregate(
            lambda image_mses: dict(image_mse=torch.stack(image_mses).mean().item())
        ),
        eps_mse=lantern.Metric().aggregate(
            lambda eps_mses: dict(eps_mse=torch.stack(eps_mses).mean().item())
        ),
    )
