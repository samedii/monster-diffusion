from tempfile import TemporaryDirectory
from pathlib import Path
from uuid import uuid4
from tqdm import tqdm
from PIL import Image
from cleanfid import fid

from monster_diffusion import data, settings


if __name__ == "__main__":
    for name, evaluate_datastream in data.evaluate_datastreams().items():
        stat_name = settings.FID_STATISTICS_NAMES[name]
        if fid.test_stats_exists(stat_name, mode="clean"):
            fid.remove_custom_stats(stat_name)

        with TemporaryDirectory() as dataset_dir:
            evaluate_datastreams = data.evaluate_datastreams()
            for example in tqdm(evaluate_datastream):
                Image.fromarray(example.image).save(
                    Path(dataset_dir) / f"{uuid4()}.png"
                )

            fid.make_custom_stats(stat_name, dataset_dir)
