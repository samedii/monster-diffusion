import argparse
import os
import json
from pathlib import Path
import torch
import torch.utils.tensorboard
import torchvision.transforms.functional as TF
import lantern
from lantern import set_seeds
import nicefid

import data
from monster_diffusion import settings
from monster_diffusion import metrics
from monster_diffusion.log_examples import log_examples
from monster_diffusion.model.model import Model
from monster_diffusion.generate_samples import generate_samples
from monster_diffusion.generate_cheat_samples import generate_cheat_samples
from monster_diffusion.tools.seeded_randn import seeded_randn
from monster_diffusion.model.variational_encoder import VariationalEncoderLDM


def evaluate(config):
    set_seeds(config["seed"])
    device = torch.device("cuda" if config["use_cuda"] else "cpu")
    torch.set_grad_enabled(False)

    model = Model().eval().to(device)
    # variational_encoder = VariationalEncoderLDM().eval().to(device)

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
            "train",
            "early_stopping",
        ]
    }

    print("Loading model checkpoint")
    model.load_state_dict(torch.load("model/average_model.pt", map_location=device))
    # variational_encoder.load_state_dict(
    #     torch.load("model/variational_encoder.pt", map_location=device)
    # )

    tensorboard_logger = torch.utils.tensorboard.SummaryWriter(log_dir="tb")

    global_step = 0

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
            global_step,
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

            evaluation_ts = Model.evaluation_ts()
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
            # with lantern.module_eval(variational_encoder):
            #     variational_features = variational_encoder.features(images, noise)

            diffused = model.diffuse(images, ts, noise)
            with lantern.module_eval(model):
                predictions = model.predictions(
                    diffused,
                    ts,
                    nonleaky_augmentations,
                    # variational_features,
                )
                # variational_losses = variational_features.losses()
                # variational_loss = variational_losses.mean()
                loss = predictions.loss(images, noise)

            evaluate_metrics["loss"].update_(loss)
            # evaluate_metrics["variational_loss"].update_(variational_loss)
            evaluate_metrics["image_mse"].update_(predictions.image_mse(images))
            evaluate_metrics["eps_mse"].update_(predictions.eps_mse(noise))

        for metric in evaluate_metrics.values():
            metric.log_dict(tensorboard_logger, name, global_step)

        print(lantern.MetricTable(name, evaluate_metrics))
        log_examples(tensorboard_logger, name, global_step, examples, predictions)

    def generator(data_loader):
        for examples in data_loader:
            yield torch.stack(
                [TF.to_tensor(example.image).div(255) for example in examples]
            )

    reference_features = {
        name: nicefid.Features.from_iterator(generator(data_loader))
        for name, data_loader in evaluate_data_loaders.items()
    }

    for n_evaluations in [20, 100]:
        generated_features = nicefid.Features.from_iterator(
            generate_samples(model, n_samples=1024 * 10, n_evaluations=n_evaluations)
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
                f"{name}/fid@{n_evaluations}", fid_score, global_step
            )
            print(f"{name}/fid@{n_evaluations}: {fid_score}")

        # cheat_fid_scores = {
        #     name: nicefid.compute_fid(
        #         features,
        #         nicefid.Features.from_iterator(
        #             generate_cheat_samples(
        #                 model,
        #                 variational_encoder,
        #                 evaluate_data_loaders[name],
        #                 n_evaluations=n_evaluations,
        #             )
        #         ),
        #     )
        #     for name, features in reference_features.items()
        # }
        # for name, cheat_fid_score in cheat_fid_scores.items():
        #     tensorboard_logger.add_scalar(
        #         f"{name}/cheat_fid@{n_evaluations}",
        #         cheat_fid_score,
        #         global_step,
        #     )
        #     print(f"{name}/cheat_fid@{n_evaluations}: {cheat_fid_score}")

    tensorboard_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate_batch_size", type=int, default=32)
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
