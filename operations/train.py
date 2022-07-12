import argparse
import os
import json
from pathlib import Path
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.utils.tensorboard
import torchvision.transforms.functional as TF
import lantern
from lantern import set_seeds, worker_init_fn
import nicefid
import wandb

import data
from monster_diffusion.model.model import Model
from monster_diffusion import metrics
from monster_diffusion.log_examples import log_examples
from monster_diffusion.model.variational_encoder import VariationalEncoderLDM
from monster_diffusion.generate_samples import generate_samples
from monster_diffusion.generate_cheat_samples import generate_cheat_samples
from monster_diffusion.tools.seeded_randn import seeded_randn
from monster_diffusion.tools.inverse_lr import InverseLR
from monster_diffusion.kl_weight_controller import KLWeightController
from monster_diffusion.model.ema import EMAWarmup, ema_update
from monster_diffusion import settings


def train(config):
    set_seeds(config["seed"])
    device = torch.device("cuda" if config["use_cuda"] else "cpu")
    torch.set_grad_enabled(False)

    model = Model().eval().to(device)
    model_ema = deepcopy(model).eval()
    variational_encoder = VariationalEncoderLDM().eval().to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(variational_encoder.parameters()),
        lr=config["learning_rate"],
        betas=(0.95, 0.999),
        eps=1e-6,
        weight_decay=1e-3,
    )
    scheduler = InverseLR(optimizer, inv_gamma=20000, power=1, warmup=0.99)
    ema_scheduler = EMAWarmup(power=0.6667, max_value=0.9999)
    kl_weight_controller = KLWeightController(
        weights=[0.001],
        targets=[config["kl_target"]],
    )

    train_datastream = data.train_datastream()

    train_data_loader = train_datastream.data_loader(
        batch_size=config["batch_size"],
        n_batches_per_epoch=config["evaluate_every"],
        collate_fn=list,
        num_workers=config["n_workers"],
        worker_init_fn=worker_init_fn(config["seed"]),
        persistent_workers=(config["n_workers"] >= 1),
    )

    evaluate_datastreams = data.evaluate_datastreams()
    evaluate_data_loaders = {
        f"evaluate_{name}": (
            (
                evaluate_datastreams[name].data_loader(
                    batch_size=config["evaluate_batch_size"],
                    collate_fn=list,
                    num_workers=config["n_workers"],
                )
            )
        )
        for name in [
            # "train",
            "early_stopping",
        ]
    }

    def generator(data_loader):
        for examples in data_loader:
            yield torch.stack(
                [TF.to_tensor(example.image).div(255) for example in examples]
            )

    reference_features = {
        name: nicefid.Features.from_iterator(generator(data_loader))
        for name, data_loader in evaluate_data_loaders.items()
    }

    if Path("model").exists():
        print("Loading model checkpoint")
        model.load_state_dict(torch.load("model/model.pt", map_location=device))
        model_ema.load_state_dict(torch.load("model/model_ema.pt", map_location=device))
        variational_encoder.load_state_dict(
            torch.load("model/variational_encoder.pt", map_location=device)
        )
        optimizer.load_state_dict(torch.load("model/optimizer.pt", map_location=device))
        lantern.set_learning_rate(optimizer, config["learning_rate"])

    tensorboard_logger = torch.utils.tensorboard.SummaryWriter(log_dir="tb")
    early_stopping = lantern.EarlyStopping(tensorboard_logger=tensorboard_logger)
    train_metrics = metrics.train_metrics()

    n_train_steps = 0
    for _ in lantern.Epochs(config["max_steps"] // config["evaluate_every"]):

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
            diffused = model.diffuse(images, ts, noise)
            with lantern.module_train(model), lantern.module_train(
                variational_encoder
            ), torch.enable_grad():
                variational_features = variational_encoder.features_(images, noise)
                predictions = model.predictions_(
                    diffused,
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
            scheduler.step()
            ema_decay = ema_scheduler.get_value()
            ema_update(model, model_ema, ema_decay)
            ema_scheduler.step()

            if n_train_steps >= 100:
                kl_weight_controller.update_([variational_loss])

            n_train_steps += 1

            tensorboard_logger.add_scalar(
                "learning_rate", scheduler.get_lr()[0], n_train_steps
            )
            tensorboard_logger.add_scalar(
                "variational_weight", kl_weight_controller.weights[0], n_train_steps
            )
            tensorboard_logger.add_scalar(
                "kl_target", kl_weight_controller.targets[0], n_train_steps
            )

            train_metrics["loss"].update_(loss)
            train_metrics["variational_loss"].update_(variational_loss)
            train_metrics["image_mse"].update_(predictions.image_mse(images))
            train_metrics["eps_mse"].update_(predictions.eps_mse(noise))

            for metric in train_metrics.values():
                metric.log_dict(tensorboard_logger, "train", n_train_steps)

        print(lantern.MetricTable("train", train_metrics))
        log_examples(tensorboard_logger, "train", n_train_steps, examples, predictions)

        for n_evaluations in [20, 100, 1000]:
            tensorboard_logger.add_images(
                f"samples/@{n_evaluations}",
                torch.cat(
                    [
                        image.clamp(0, 1)
                        for index, image in enumerate(
                            model.sample(5, n_evaluations, progress=True)
                        )
                        if (index + 1) % (n_evaluations // 10) == 0
                    ],
                    dim=-2,
                ),
                n_train_steps,
                dataformats="NCHW",
            )

        for name, data_loader in evaluate_data_loaders.items():
            evaluate_metrics = metrics.evaluate_metrics()
            for examples in lantern.ProgressBar(data_loader, name):
                images = torch.stack(
                    [TF.to_tensor(example.image) for example in examples]
                ).div(255)
                nonleaky_augmentations = torch.stack(
                    [
                        torch.from_numpy(example.nonleaky_augmentations)
                        for example in examples
                    ]
                )

                evaluation_ts = model_ema.evaluation_ts()
                choice = torch.tensor(
                    [example.hash() % len(evaluation_ts) for example in examples]
                )
                ts = evaluation_ts[choice]
                noise = torch.stack(
                    [
                        seeded_randn(settings.INPUT_SHAPE, abs(example.hash()))
                        for example in examples
                    ]
                )

                with lantern.module_eval(variational_encoder):
                    variational_features = variational_encoder.features(images, noise)

                diffused = model_ema.diffuse(images, ts, noise)
                with lantern.module_eval(model_ema):
                    predictions = model_ema.predictions(
                        diffused,
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

                evaluate_metrics["loss"].update_(loss)
                evaluate_metrics["variational_loss"].update_(variational_loss)
                evaluate_metrics["image_mse"].update_(predictions.image_mse(images))
                evaluate_metrics["eps_mse"].update_(predictions.eps_mse(noise))

            for metric in evaluate_metrics.values():
                metric.log_dict(tensorboard_logger, name, n_train_steps)

            print(lantern.MetricTable(name, evaluate_metrics))
            log_examples(tensorboard_logger, name, n_train_steps, examples, predictions)

        n_evaluations = 100
        generated_features = nicefid.Features.from_iterator(
            generate_samples(model, n_evaluations=n_evaluations)
        )
        fid_scores = {
            name: nicefid.compute_fid(
                features,
                generated_features,
            )
            for name, features in reference_features.items()
        }
        for name, fid_score in fid_scores.items():
            tensorboard_logger.add_scalar(
                f"{name}/fid@{n_evaluations}", fid_score, n_train_steps
            )
            print(f"{name}/fid@{n_evaluations}: {fid_score}")

        n_evaluations = 20
        cheat_fid_scores = {
            name: nicefid.compute_fid(
                features,
                nicefid.Features.from_iterator(
                    generate_cheat_samples(
                        model,
                        variational_encoder,
                        evaluate_data_loaders[name],
                        n_evaluations=n_evaluations,
                    )
                ),
            )
            for name, features in reference_features.items()
        }
        for name, cheat_fid_score in cheat_fid_scores.items():
            tensorboard_logger.add_scalar(
                f"{name}/cheat_fid@{n_evaluations}",
                cheat_fid_score,
                n_train_steps,
            )
            print(f"{name}/cheat_fid@{n_evaluations}: {cheat_fid_score}")

        early_stopping = early_stopping.score(-fid_scores["evaluate_early_stopping"])
        if early_stopping.scores_since_improvement == 0:
            torch.save(model.state_dict(), "model.pt")
            torch.save(variational_encoder.state_dict(), "variational_encoder.pt")
            torch.save(optimizer.state_dict(), "optimizer.pt")
            torch.save(kl_weight_controller.state_dict(), "kl_weight_controller.pt")
        elif early_stopping.scores_since_improvement > config["patience"]:
            break
        early_stopping.log(n_train_steps).print()

    tensorboard_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--evaluate_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--kl_target", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=5000 * 200)
    parser.add_argument("--evaluate_every", default=5000, type=int)
    parser.add_argument("--patience", type=float, default=10)
    parser.add_argument("--n_workers", default=8, type=int)
    parser.add_argument("--debug", default=0, type=int)
    args = parser.parse_args()

    config = vars(args)
    config.update(
        seed=1,
        use_cuda=torch.cuda.is_available(),
        run_id=os.getenv("RUN_ID"),
    )

    Path("config.json").write_text(json.dumps(config))

    if config["debug"] == 0:
        wandb.init(
            project="monster-diffusion",
            save_code=False,
            resume="never",
            magic=True,
            anonymous="never",
            id=config["run_id"],
            name=Path(".guild/attrs/label").read_text(),
            tags=[config["run_id"]],
            sync_tensorboard=True,
            config=config,
        )

    train(config)
