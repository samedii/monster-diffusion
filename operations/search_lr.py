import argparse
import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.utils.tensorboard
import lantern
from lantern import set_seeds, worker_init_fn

import data
from monster_diffusion.model.model import Model
from monster_diffusion import metrics
from monster_diffusion.model.variational_encoder import VariationalEncoderLDM


def train(config):
    set_seeds(config["seed"])
    device = torch.device("cuda" if config["use_cuda"] else "cpu")
    torch.set_grad_enabled(False)

    model = Model().eval().to(device)
    variational_encoder = VariationalEncoderLDM().eval().to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(variational_encoder.parameters()),
        lr=config["start_learning_rate"],
        # weight_decay=1e-2,
    )

    train_datastream = data.train_datastream()

    train_data_loader = train_datastream.data_loader(
        batch_size=config["batch_size"],
        n_batches_per_epoch=config["n_batches"],
        collate_fn=list,
        num_workers=config["n_workers"],
        worker_init_fn=worker_init_fn(config["seed"]),
        persistent_workers=(config["n_workers"] >= 1),
    )

    if Path("model").exists():
        print("Loading model checkpoint")
        model.load_state_dict(torch.load("model/model.pt", map_location=device))
        variational_encoder.load_state_dict(
            torch.load("model/variational_encoder.pt", map_location=device)
        )
        optimizer.load_state_dict(torch.load("model/optimizer.pt", map_location=device))
        lantern.set_learning_rate(optimizer, config["start_learning_rate"])

    tensorboard_logger = torch.utils.tensorboard.SummaryWriter(log_dir="tb")
    train_metrics = metrics.train_metrics()

    lr_multiplier = (config["stop_learning_rate"] / config["start_learning_rate"]) ** (
        1 / config["n_batches"]
    )
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer,
        lr_lambda=lambda _: lr_multiplier,
    )

    n_train_steps = 0

    for examples in lantern.ProgressBar(train_data_loader, "train", train_metrics):
        images = torch.from_numpy(np.stack([example.image for example in examples]))

        ts = Model.training_ts(len(examples))
        noise = torch.randn_like(images.float())
        diffused = model.diffuse(images, ts, noise)
        with lantern.module_train(model), lantern.module_train(
            variational_encoder
        ), torch.enable_grad():
            variational_features = variational_encoder.features_(images, noise)
            predictions = model.predictions_(
                diffused,
                ts,
                variational_features,
            )
            loss = predictions.loss(examples, noise)
            variational_loss = variational_features.loss()
            loss = (
                predictions.loss(examples, noise)
                + config["kl_weight"]
                * torch.relu(variational_loss - config["kl_target"]) ** 2
            )
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        n_train_steps += 1

        scheduler.step()
        tensorboard_logger.add_scalar(
            "learning_rate", scheduler.get_lr()[0], n_train_steps
        )

        train_metrics["loss"].update_(loss)
        train_metrics["variational_loss"].update_(variational_loss)
        train_metrics["image_mse"].update_(predictions.image_mse(examples))
        train_metrics["eps_mse"].update_(predictions.eps_mse(noise))

        for metric in train_metrics.values():
            metric.log_dict(tensorboard_logger, "train", n_train_steps)

    tensorboard_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--start_learning_rate", type=float, default=1e-8)
    parser.add_argument("--stop_learning_rate", type=float, default=1e-1)
    parser.add_argument("--kl_target", type=float, default=1e-3)
    parser.add_argument("--n_batches", default=1000, type=int)
    parser.add_argument("--patience", type=float, default=4)
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
