import argparse
import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.utils.tensorboard
import lantern
from lantern import set_seeds, worker_init_fn
from cleanfid import fid

from monster_diffusion import (
    data,
    Model,
    metrics,
    log_examples,
    settings,
)
from monster_diffusion.temporary_samples import temporary_samples
from monster_diffusion.temporary_cheat_samples import temporary_cheat_samples
from monster_diffusion.tools.seeded_randn import seeded_randn
from monster_diffusion.model.variational_encoder import VariationalEncoderLDM


def evaluate(config):
    set_seeds(config["seed"])
    device = torch.device("cuda" if config["use_cuda"] else "cpu")
    torch.set_grad_enabled(False)

    model = Model().eval().to(device)
    variational_encoder = VariationalEncoderLDM().eval().to(device)

    evaluate_datastreams = data.evaluate_datastreams()
    evaluate_data_loaders = {
        f"evaluate_{name}": (
            (
                evaluate_datastreams[name].data_loader(
                    batch_size=config["eval_batch_size"],
                    collate_fn=list,
                    num_workers=config["n_workers"],
                )
            )
        )
        for name in [
            "train",
            "early_stopping",
        ]
    }

    print("Loading model checkpoint")
    model.load_state_dict(torch.load("model/model.pt", map_location=device))
    variational_encoder.load_state_dict(
        torch.load("model/variational_encoder.pt", map_location=device)
    )

    tensorboard_logger = torch.utils.tensorboard.SummaryWriter(log_dir="tb")

    epoch = 0

    for n_evaluations in [20, 100, 1000]:
        tensorboard_logger.add_images(
            f"samples/@{n_evaluations}",
            np.uint8(
                np.stack(
                    [
                        image.clamp(0, 1).cpu().numpy()
                        for index, image in enumerate(
                            model.elucidated_sample(1, n_evaluations, progress=True)
                        )
                        if (index + 1) % (n_evaluations // 10) == 0
                    ]
                )
                * 255
            ),
            epoch,
            dataformats="NCHW",
        )

    for name, data_loader in evaluate_data_loaders.items():
        evaluate_metrics = metrics.evaluate_metrics()
        for examples in lantern.ProgressBar(data_loader, name):
            images = torch.from_numpy(np.stack([example.image for example in examples]))

            evaluation_ts = Model.evaluation_ts()
            choice = torch.tensor(
                [example.hash() % len(evaluation_ts) for example in examples]
            )
            ts = evaluation_ts[choice]
            noise = torch.stack(
                [
                    seeded_randn(example.image.shape, abs(example.hash()))
                    for example in examples
                ]
            )
            with lantern.module_eval(variational_encoder):
                variational_features = variational_encoder.features(images, noise)

            diffused = model.diffuse(images, ts, noise)
            with lantern.module_eval(model):
                predictions = model.predictions(
                    diffused,
                    ts,
                    variational_features,
                )
                loss = predictions.loss(examples, noise)
                variational_loss = variational_features.loss()

            evaluate_metrics["loss"].update_(loss)
            evaluate_metrics["variational_loss"].update_(variational_loss)
            evaluate_metrics["image_mse"].update_(predictions.image_mse(examples))
            evaluate_metrics["eps_mse"].update_(predictions.eps_mse(noise))

        for metric in evaluate_metrics.values():
            metric.log_dict(tensorboard_logger, name, epoch)

        print(lantern.MetricTable(name, evaluate_metrics))
        log_examples(tensorboard_logger, name, epoch, examples, predictions)

    for n_evaluations in [20]:
        with temporary_cheat_samples(
            model,
            variational_encoder,
            evaluate_data_loaders["evaluate_early_stopping"],
            n_evaluations=n_evaluations,
        ) as samples_dir:
            cheat_fid_scores = {
                name: fid.compute_fid(
                    samples_dir,
                    dataset_name=settings.FID_STATISTICS_NAMES[name],
                    dataset_res=96,
                    dataset_split="custom",
                    verbose=False,
                    num_workers=0,
                )
                for name in ["train", "early_stopping"]
            }
            for name, cheat_fid_score in cheat_fid_scores.items():
                tensorboard_logger.add_scalar(
                    f"evaluate_{name}/cheat_fid@{n_evaluations}", cheat_fid_score, epoch
                )
                print(f"evaluate_{name}/cheat_fid@{n_evaluations}: {cheat_fid_score}")

        with temporary_samples(
            model, batch_size=config["eval_batch_size"], n_evaluations=n_evaluations
        ) as samples_dir:
            fid_scores = {
                name: fid.compute_fid(
                    samples_dir,
                    dataset_name=settings.FID_STATISTICS_NAMES[name],
                    dataset_res=96,
                    dataset_split="custom",
                    verbose=False,
                    num_workers=config["n_workers"],
                )
                for name in ["train", "early_stopping"]
            }
            for name, fid_score in fid_scores.items():
                tensorboard_logger.add_scalar(
                    f"evaluate_{name}/fid@{n_evaluations}", fid_score, epoch
                )
                print(f"evaluate_{name}/fid@{n_evaluations}: {fid_score}")

    tensorboard_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--n_workers", default=8, type=int)
    args = parser.parse_args()

    config = vars(args)
    config.update(
        seed=1,
        use_cuda=torch.cuda.is_available(),
        run_id=os.getenv("RUN_ID"),
    )

    Path("config.json").write_text(json.dumps(config))

    evaluate(config)
