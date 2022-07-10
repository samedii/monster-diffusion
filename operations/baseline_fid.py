from tempfile import TemporaryDirectory
from pathlib import Path
from uuid import uuid4
from tqdm import tqdm
from PIL import Image
from monster_diffusion import data
from cleanfid import fid

from monster_diffusion import settings


if __name__ == "__main__":
    evaluate_datastream = data.evaluate_datastreams()["early_stopping"]

    with TemporaryDirectory() as dataset_dir:
        for example in tqdm(evaluate_datastream):
            Image.fromarray(example.image).save(Path(dataset_dir) / f"{uuid4()}.png")

        fid_score = fid.compute_fid(
            dataset_dir,
            dataset_name=settings.FID_STATISTICS_NAMES["train"],
            dataset_res=settings.INPUT_HEIGHT,
            dataset_split="custom",
        )
    print(f"fid score between train and early stopping: {fid_score}")
