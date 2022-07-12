import argparse
from contextlib import contextmanager
import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
import torchvision.transforms.functional as TF
import lantern
from lantern import set_seeds, worker_init_fn

import data
from monster_diffusion.model.model import Model
from monster_diffusion.kl_weight_controller import KLWeightController
from monster_diffusion import metrics
from monster_diffusion.log_examples import log_examples
from monster_diffusion.model.variational_encoder import VariationalEncoderLDM


def train(config):
    set_seeds(config["seed"])
    device = torch.device("cuda" if config["use_cuda"] else "cpu")
    torch.set_grad_enabled(False)

    previous_config = json.loads(Path("model/config.json").read_text())

    model = Model().eval().to(device)
    variational_encoder = VariationalEncoderLDM().eval().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    kl_weight_controller = KLWeightController(
        weights=[0.001],
        targets=[previous_config["kl_target"]],
    )

    average_model = torch.optim.swa_utils.AveragedModel(model)
    average_variational_encoder = torch.optim.swa_utils.AveragedModel(
        variational_encoder
    )

    train_datastream = data.train_datastream()

    train_data_loader = train_datastream.data_loader(
        batch_size=config["batch_size"],
        n_batches_per_epoch=config["n_batches_per_epoch"],
        collate_fn=list,
        num_workers=config["n_workers"],
        worker_init_fn=worker_init_fn(config["seed"]),
        persistent_workers=(config["n_workers"] >= 1),
    )

    print("Loading model checkpoint")
    model.load_state_dict(torch.load("model/model.pt", map_location=device))
    variational_encoder.load_state_dict(
        torch.load("model/variational_encoder.pt", map_location=device)
    )
    kl_weight_controller.load_state_dict(
        torch.load("model/kl_weight_controller.pt", map_location=device)
    )

    tensorboard_logger = torch.utils.tensorboard.SummaryWriter(log_dir="tb")
    train_metrics = metrics.train_metrics()

    n_train_steps = 0
    for epoch in lantern.Epochs(config["n_epochs"]):

        for examples in lantern.ProgressBar(train_data_loader, "train", train_metrics):
            images = torch.stack(
                [TF.to_tensor(example.image) for example in examples]
            ).div(255)
            nonleaky_augmentations = torch.stack(
                [
                    torch.from_numpy(example.nonleaky_augmentations)
                    for example in examples
                ]
            )

            ts = Model.training_ts(len(examples))
            noise = torch.randn_like(images.float())
            diffused_images = model.diffuse(images, ts, noise)
            with lantern.module_train(model), lantern.module_train(
                variational_encoder
            ), torch.enable_grad():
                variational_features = variational_encoder.features_(images, noise)
                predictions = model.predictions_(
                    diffused_images,
                    ts,
                    nonleaky_augmentations,
                    variational_features,
                )
                variational_losses = variational_features.losses()
                variational_loss = variational_losses.mean()
                loss = (
                    predictions.loss(images, noise)
                    + kl_weight_controller.weights[0]
                    * F.softplus(
                        variational_losses.log()
                        - torch.tensor(kl_weight_controller.targets[0]).log(),
                        beta=5,
                    ).mean()
                )
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            n_train_steps += 1

            train_metrics["loss"].update_(loss)
            train_metrics["variational_loss"].update_(variational_loss)
            train_metrics["image_mse"].update_(predictions.image_mse(images))
            train_metrics["eps_mse"].update_(predictions.eps_mse(noise))

            for metric in train_metrics.values():
                metric.log_dict(tensorboard_logger, "train", n_train_steps)

        average_model.update_parameters(model)
        average_variational_encoder.update_parameters(variational_encoder)

        print(lantern.MetricTable("train", train_metrics))
        log_examples(tensorboard_logger, "train", epoch, examples, predictions)

    batchnorm_dataloader = train_datastream.data_loader(
        batch_size=config["evaluate_batch_size"],
        n_batches_per_epoch=10000 // config["evaluate_batch_size"],
        collate_fn=list,
        num_workers=config["n_workers"],
        worker_init_fn=worker_init_fn(config["seed"]),
        persistent_workers=(config["n_workers"] >= 1),
    )

    model.load_state_dict(average_model.module.state_dict())
    variational_encoder.load_state_dict(average_variational_encoder.module.state_dict())

    train_metrics = metrics.train_metrics()
    n_train_steps = 0
    with update_batchnorm(nn.ModuleList([model, variational_encoder])):
        for examples in lantern.ProgressBar(
            batchnorm_dataloader, "batchnorm", train_metrics
        ):
            images = torch.stack(
                [TF.to_tensor(example.image) for example in examples]
            ).div(255)
            nonleaky_augmentations = torch.stack(
                [
                    torch.from_numpy(example.nonleaky_augmentations)
                    for example in examples
                ]
            )

            ts = Model.training_ts(len(examples))
            noise = torch.randn_like(images.float())
            diffused_images = model.diffuse(images, ts, noise)
            with lantern.module_train(model), lantern.module_train(variational_encoder):
                variational_features = variational_encoder.features_(images, noise)
                predictions = model.predictions_(
                    diffused_images,
                    ts,
                    nonleaky_augmentations,
                    variational_features,
                )
                variational_losses = variational_features.losses()
                variational_loss = variational_losses.mean()
                loss = (
                    predictions.loss(images, noise)
                    + kl_weight_controller.weights[0]
                    * F.softplus(
                        variational_losses.log()
                        - torch.tensor(kl_weight_controller.targets[0]).log(),
                        beta=5,
                    ).mean()
                )

            n_train_steps += 1

            train_metrics["loss"].update_(loss)
            train_metrics["variational_loss"].update_(variational_loss)
            train_metrics["image_mse"].update_(predictions.image_mse(images))
            train_metrics["eps_mse"].update_(predictions.eps_mse(noise))

            for metric in train_metrics.values():
                metric.log_dict(tensorboard_logger, "batchnorm", n_train_steps)

    torch.save(model.state_dict(), "average_model.pt")
    torch.save(variational_encoder.state_dict(), "variational_encoder.pt")

    tensorboard_logger.close()


@contextmanager
def update_batchnorm(model):
    try:
        momenta = {}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum

        if not momenta:
            return

        was_training = model.training
        model.train()
        for module in momenta.keys():
            module.momentum = None
            module.num_batches_tracked *= 0

        yield
    finally:
        for bn_module in momenta.keys():
            bn_module.momentum = momenta[bn_module]
        model.train(was_training)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--evaluate_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_batches_per_epoch", default=50, type=int)
    parser.add_argument("--n_workers", default=8, type=int)
    args = parser.parse_args()

    config = vars(args)
    config.update(
        seed=1,
        use_cuda=torch.cuda.is_available(),
        run_id=os.getenv("RUN_ID"),
    )

    Path("config.json").write_text(json.dumps(config))

    train(config)
