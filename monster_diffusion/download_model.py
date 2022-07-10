import requests
from pathlib import Path

MODEL_DIGEST = "??"
MODEL_URL = (
    f"https://s3.wasabisys.com/???/{MODEL_DIGEST}/model.pt"
)
MODEL_PATH = Path(f"{MODEL_DIGEST}/model.pt")


def download_model():
    pass

    # if MODEL_PATH.exists():
    #     raise Exception(f"Model already downloaded: {MODEL_PATH}")
    # else:
    #     MODEL_PATH.parent.mkdir()
    #     model_response = requests.get(MODEL_URL, allow_redirects=True)

    #     if model_response.status_code == 200:
    #         MODEL_PATH.write_bytes(model_response.content)
    #     else:
    #         raise Exception(f"Error downloading model: {model_response.status_code}")


if __name__ == "__main__":
    download_model()


# def test_download_model(tmpdir):
#     global MODEL_PATH
#     MODEL_PATH = Path(tmpdir) / MODEL_DIGEST / "model.pt"
#     download_model()
