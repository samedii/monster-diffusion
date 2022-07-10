import numpy as np


def log_examples(logger, name, epoch, examples, predictions):
    logger.add_images(
        f"{name}/predictions",
        np.stack(
            [
                np.float32(predictions[index].representation(examples[index])) / 255
                for index in np.random.choice(
                    len(examples), size=min(5, len(examples)), replace=False
                )
            ]
        ),
        epoch,
        dataformats="NHWC",
    )
